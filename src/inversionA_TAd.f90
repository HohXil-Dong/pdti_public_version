Program inversionA_TAd
   !============================================================
   ! Potency density tensor inversion  (Y. Yagi)
   ! To be run after the optimal values of the hyperparameters are estimated by ABIC_new2.
   ! Ifalg is set to 1, the LBFGSB can be used.
   !  Parameter
   !     imax:  The maximun number of Waveform
   !     jmax:  The maximun number of Station
   !     i_LBFGSB : Set i_LBFGSB to 1 to use LBFGSB. (default = 0)
   !     a_low_b  : Minimum value of displacement(icm=1) (only used when i_LBFGSB=1)
   !-----------
   !     H(1:kmax,1:lmax): Kannel matrix
   !     d(1:kmax): Data vector
   !     a(1:lmax): Solution vecoter
   !     G1(1:lmax,1:lmax) : time and space constrain matrix
   !     Cm(1:kmax,1:kmax) : Covariance matrix of data
   !     d_* : double precision
   !  Estation Error problem
   !     The estimation error in the output seems to be not meaningful
   !     because the non-diagonal components of the model covariance matrix are ignored.
   !     The calculation was changed so that estimation errors are not calculated.
   !============================================================
!  use, non_intrinsic:: optimize_lib, only: nnls_lbfgsb
   !############################################################################
   !#                                                                          #
   !#       [NEW FEATURE] OUTPUT SYNTHETIC AND OBSERVED WAVEFORMS              #
   !#                                                                          #
   !#  This section is NEW compared to inversionA_TAd_old.f90, which only      #
   !#  computed the residual without outputting waveforms for visualization.   #
   !#                                                                          #
   !#  Added Features:                                                         #
   !#    1. Downsampled waveforms (.syn / .obs) - same dt as inversion         #
   !#    2. High-resolution waveforms (.syn01 / .obs01) - fixed dt=0.1s        #
   !#                                                                          #
   !############################################################################
   implicit none
   integer,parameter  :: imax=2501,jmax=201
   integer,parameter  :: i_LBFGSB=0
   real,parameter     :: a_low_b=-0.2
   !  Inversion Parameter
   real,allocatable   :: H(:,:), d(:), a(:), siga(:)
   real,allocatable   :: G1(:,:)
   real, allocatable  :: grn_m(:,:,:),obs_m(:,:,:)
   double precision, allocatable :: d_HtH(:,:), d_Htd(:), d_a(:), d_siga(:)
   double precision, allocatable :: d_H(:,:), d_d(:), d_Cm(:,:)
   double precision   :: d_btb, d_det_A
   !  Station Parameter
   character stcd(jmax)*10,comp(jmax)*4,title*50,cmode*1
   real,allocatable ::  Wobs(:,:)
   real cp(jmax),t1(jmax),dt(jmax),sigw(jmax),sigwd(jmax),cp1(jmax),tlength(jmax)
   real az(jmax),del(jmax)
   integer ndj(jmax)
   !  ABIC Parameter
   real beta(3),ABIC
   real,allocatable :: tr(:,:)
   integer,allocatable :: jtn(:,:),l_id_m(:,:,:,:)
   !  working space
   real,allocatable :: zz1(:),zz2(:)
   !-------
   integer :: i,j,k,l,n,l1,l2
   integer :: kmax,lmax,jn,ndmax,nmodel,ndata
   integer :: mn,nn,m0,n0,icmn
   integer :: jtn0,itera,nfabic,nsurface,nsurface_o,nflag
   integer :: nwk
   real    :: xx,yy,vr0,ylength,depth
   real    :: strike,dip,slip,rslip,rtime,st_max,r_s_t,cr
   real    :: shift,para1,para2,para3
   real    :: alpha1,alpha2,dump,s,var
   real    :: p_sec
   real    :: wk1,wk2,f_o,f_g
   real    :: det_Cm
   real    :: f_nrom2
   !============================================================================
   !  [NEW] Variables for Waveform Output (not in inversionA_TAd_old.f90)
   !============================================================================
   !
   !  These variables are used ONLY for outputting synthetic and observed
   !  waveforms to wave.syn/ directory. They do not affect inversion results.
   !
   !----------------------------------------------------------------------------
   !  0.1s Synthetic Waveform Variables
   !----------------------------------------------------------------------------
   real, parameter :: dts = 0.1              ! Output sampling interval (s)
   integer :: ndjs                           ! Number of output samples
   integer :: m, jt, icm                     ! Loop indices: knot, time, basis
   real, allocatable :: syn_s(:)             ! Accumulated synthetic waveform
   real, allocatable :: Umn_s(:)             ! Resampled single-source response
   real, allocatable :: green(:)             ! Convolved Green's function
   real, allocatable :: so(:)                ! Source time function
   real, allocatable :: gb(:,:,:,:)          ! Raw Green's functions (ndg,mn,nn,icmn)
   integer :: ic1, ic2                       ! String index helpers
   integer :: ndg, iw                        ! GF length, convolution length
   real :: tg0, dtg                          ! GF origin time, GF sampling
   real :: work, val, tmax                   ! Temporary variables
   character :: cha*1                        ! Basis index character ('1'-'5')
   !----------------------------------------------------------------------------
   !  0.1s Observed Waveform Variables
   !----------------------------------------------------------------------------
   integer :: nstcd_obs                      ! Station code string length
   integer :: nd0_obs, nend_obs, np0_obs     ! Raw obs length, PP index, pre-P index
   real :: t1_tmp                            ! P-wave arrival from obs header
   real :: dt0_obs                           ! Raw observation sampling
   real :: sig_obs                           ! Observation noise level
   real :: shift_obs                         ! Time shift for resampling
   real :: junk1, junk2, junk3               ! Unused header fields
   real, allocatable :: wv_obs(:)            ! Raw observation waveform
   real, allocatable :: wvv_obs(:)           ! Resampled observation waveform
   !============================================================================
   !+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   !  Read Input Parameter
   read(5,'(a50)')title
   read(5,*)rtime,jtn0,vr0,shift,para1,para2,para3,st_max, r_s_t, itera, cr
   read(5,*)(beta(i),i=1,3),nfabic,nsurface_o,nflag,cmode,alpha1,alpha2,dump,s
   do j=1,jmax
      read(5,*,err=998,end =999)stcd(j),comp(j),cp(j),sigw(j),tlength(j),dt(j)
      cp(j)=cp(j)+shift
      jn = j
998   continue
   end do
999 continue
   !+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   open(10,file="fault.dat")
   read(10,*)xx,yy,mn,nn,m0,n0,icmn,depth,rslip,nsurface,ylength,strike,dip,slip
   close(10)
   if(icmn.eq.1) rslip = 0.
   !---
   allocate(Wobs(imax,jn))
   !p_sec =  get_pre_sec(tlength,jn)
   open(60,file=".station.abic")
   read(60,*) jn,p_sec
   call readOBS_f(jn,imax,stcd,comp,Wobs,t1,dt,ndj,sigw,sigwd,tlength,p_sec)
   do j=1,jn
      read(60,*) stcd(j),comp(j),dt(j),sigw(j),sigwd(j),ndj(j)
   enddo
   close(60)
   !+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   !---
   allocate(tr(mn,nn))
   call getTR(vr0,xx,yy,mn,nn,m0,n0,tr,rtime)   !Tr(m,n); Start time for each knot
   allocate(jtn(mn,nn),l_id_m(mn,nn,jtn0,icmn))
   call getJTN(jtn,tr,mn,nn,jtn0,rtime,st_max)
   call get_l_id(mn,nn,jtn,jtn0,icmn,l_id_m)
   call get_stinfo(jn,stcd,comp,az,del)
   !----
   open(10,file="d_H.matrix",form='unformatted')
   read(10) ndata,nmodel
   kmax = ndata
   lmax = nmodel
   allocate(H(kmax,lmax),d(kmax))
   do k = 1, ndata
      read(10)d(k) !,(H(k,l),l=1,nmodel)
   enddo
   close(10)
   call r_green_H(mn,nn,jtn,jtn0,l_id_m,icmn,tr,rtime,jn,stcd,comp,dt,ndj,cp,H,kmax)
   !+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   ndmax = maxval(ndj(1:jn))
   allocate(grn_m(ndmax,ndmax,jn),obs_m(ndmax,ndmax,jn))
   allocate(d_H(kmax,lmax), d_d(kmax), d_Cm(kmax,kmax))
   open(15,file="Cg.matrix",form='unformatted')
   do j=1,jn
      call get_covariance_obs_r1(obs_m(1:ndmax,1:ndmax,j),dt(j),ndmax,stcd(j),comp(j))
      read(15)nwk,nwk
      read(15)((grn_m(l1,l2,j),l1=1,ndj(j)),l2=1,ndj(j))
   enddo
   close(15)
   call correction_obs_m(ndata,d,obs_m,ndmax,jn,ndj,sigw,sigwd,f_o)
   call correction_grn_m(ndata,d,grn_m,ndmax,jn,ndj,f_g)
   call get_dCm(alpha2, obs_m, grn_m, ndmax, ndj, jn, d_Cm, kmax, det_cm)   !eq16-YF2011
   d_H = dble(H)
   d_d = dble(d)
   allocate(d_HtH(lmax,lmax), d_Htd(lmax))
   call get_HtH_Htd_dtd_d(kmax, lmax, d_H, d_d, d_Cm, d_HtH, d_Htd, d_btb)
   deallocate(d_H, d_d, d_Cm)
   deallocate(grn_m,obs_m)
   allocate(G1(lmax,lmax))
   call get_G12_matrix(mn,nn,jtn,icmn,jtn0,vr0,xx,yy,tr,l_id_m,rtime,st_max,r_s_t,itera,cr,nsurface,nsurface_o,G1,lmax)
!  open(13,file="G_12.matrix",form='unformatted')
!  do l1 = 1, nmodel
!    read(13)(G1(l1,l2),l2=l1,nmodel)
!  enddo
!  close(13)
   !+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
!  call symetric_m(lmax,G1,lmax)
   !$omp parallel do
   do l=1,lmax
      d_HtH(1:lmax,l) = d_HtH(1:lmax,l) + dble(G1(1:lmax,l)*beta(1))
   enddo
   !$omp end parallel do
   deallocate(G1)
   !+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   allocate(a(lmax),siga(lmax))
   allocate(d_a(lmax) )
   if( i_LBFGSB == 1 ) then
!    d_a(:) = nnls_lbfgsb(d_HtH, d_Htd, 5, 1.0d+7, 1.0d-5,dble(a_low_b))
      siga = 0.
   else
      call get_sol_posv_d(lmax, d_HtH, lmax, d_Htd, d_a, d_det_A)   !eq29-YF2011
      allocate(d_siga(lmax))
!----------------
!    Since estimation errors that do not take into account the covariance component
!    of the model variance shoud be not meaningful, the calculation is skipped.
!    call get_sigx_LU_d(lmax,d_HtH,lmax,d_siga)
!    siga = d_siga
      siga = 0.
!----------------
      deallocate(d_siga)
   endif
   a = d_a
   siga = sqrt(siga * (s/ndata))             ! It's not a good approximation.
   !-----------------------------------------
   !  allocate(d_x(lmax) )
   !  call get_sol_svd_d(lmax, d_HtH, lmax, d_Htd,d_x,1.0d-07)
   !  x = d_x
   !  siga = 0.
   !  deallocate(d_x)
   !-----------------------------------------
!  if( dump  /= 0. ) call get_dump_sol(a,nmodel,dump)    !eq32-YF2011
   open(16,file="x.vector",form='unformatted')
   write(16)(a(n),n=1,nmodel)
   close(16)
   wk1 = f_nrom2(ndata,d)
   allocate(zz1(max(lmax,kmax)),zz2(max(lmax,kmax)))
!  call multi_ab(ndata,nmodel,H,kmax,a,zz1)
   call SGEMV('N', ndata, nmodel, 1.0, H, kmax, a, 1, 0.0, zz1, 1)

   !============================================================================
   ! PART 1: Downsampled Waveforms (Same Resolution as Inversion)
   !============================================================================
   !
   ! Calculation Logic:
   !   - Synthetic: zz1(k) = H(k,l) * a(l), computed via SGEMV at Line 191
   !   - Observed:  d(k) is the data vector read from d_H.matrix (Line 118)
   !
   ! Data Structure:
   !   - Data vector d(1:ndata) stores all stations sequentially:
   !       d = [sta1_t1, sta1_t2, ..., sta1_tN1, sta2_t1, ..., sta_jn_tNjn]
   !   - Index k iterates through all data points across all stations
   !
   ! Time Axis:
   !   - t = (i-1) * dt(j), starting from t=0 (P-wave arrival - p_sec)
   !   - Consistent with inversion kernel H constructed in sub.r_green_H.f90
   !
   ! Reference Code:
   !   - sub.r_OBS.f90::readOBS_f() - observation preprocessing
   !   - sub.r_OBS.f90::vector_d()  - data vector assembly
   !
   !----------------------------------------------------------------------------
   call system('mkdir -p wave.syn')
   k = 0
   do j = 1, jn
      write(title, '(a, a, ".syn")') trim(stcd(j)), trim(comp(j))
      open(80, file='wave.syn/'//trim(title))

      write(title, '(a, a, ".obs")') trim(stcd(j)), trim(comp(j))
      open(81, file='wave.syn/'//trim(title))

      do i = 1, ndj(j)
         k = k + 1
         write(80, '(f12.4, 1x, e15.7)') (i-1)*dt(j), zz1(k)
         write(81, '(f12.4, 1x, e15.7)') (i-1)*dt(j), d(k)
      end do
      close(80)
      close(81)
   end do
   write(6, *) 'Synthetic and Observed waveforms saved to wave.syn/ directory.'

   !============================================================================
   ! PART 2: High-Resolution Waveforms (dt = 0.1s)
   !============================================================================
   !
   ! Purpose:
   !   Generate smooth waveforms for publication-quality figures.
   !   Original Green's functions are stored at high sampling rate (dtg),
   !   so we can reconstruct synthetics at any desired resolution.
   !
   ! Calculation Logic (following sub.r_green_H.f90):
   !   1. Read raw Green's function gb(1:ndg, m, n, icm) from wave.grn/
   !   2. Convolve with source time function: green = conv(gb, so)
   !      - sub.c_wave_lib.f90::stime_m() generates triangular STF
   !      - sub.lapack_inv.f90::conv_y() performs convolution
   !   3. Resample and time-shift to observation time frame:
   !      - shift = Tr(m,n) + rtime*(jt-1) + cp(j) + tg0
   !      - sub.c_wave_lib.f90::resample_shift() handles interpolation
   !   4. Accumulate: syn_s = sum over (m,n,jt,icm) of a(l) * Umn_s
   !   5. Apply taper_tail for consistency with inversion kernel (Line 71 in
   !      sub.r_green_H.f90)
   !
   ! Difference from Kernel Construction:
   !   - Kernel H uses inversion sampling dt(j), here we use dts=0.1s
   !   - Both apply identical time shifts and source time function
   !
   !----------------------------------------------------------------------------
   write(6,*) "Calculating 0.1s synthetics..."

   do j = 1, jn
      ! Determine output length: same time window as inversion + 10s buffer
      ndjs = nint( (ndj(j)*dt(j)) / dts ) + 100

      ! Read Green's function header to get sampling info (dtg, ndg)
      ! File format defined in GreenPointSources.f90
      ic1=index(stcd(j),' ') - 1 ; ic2 =index(comp(j),' ') - 1
      open(21,file='wave.grn/'//stcd(j)(1:ic1)//comp(j)(1:ic2)//cha(1), form='unformatted')
      read(21);read(21);read(21)
      read(21)tg0,dtg,work,ndg
      close(21)
      iw = ndg*2   ! Convolution output length

      allocate(syn_s(ndjs), Umn_s(ndjs*2), green(iw), so(iw), gb(iw,mn,nn,icmn))
      syn_s = 0.
      gb = 0.

      ! Read all Green's functions for this station (all basis components)
      ! icmn = number of moment tensor basis (1, 2, or 5)
      do icm=1,icmn
         open(20,file='wave.grn/'//stcd(j)(1:ic1)//comp(j)(1:ic2)//cha(icm), form='unformatted')
         read(20);read(20);read(20)
         read(20)tg0,dtg,work,ndg,del(j),az(j)
         ! Generate source time function (triangular, half-width = rtime)
         ! Reference: sub.c_wave_lib.f90::stime_m()
         call stime_m(so,iw,dtg,rtime)
         do n=1,nn;do m=1,mn
               read(20)(gb(i,m,n,icm),i=1,ndg)
            enddo; enddo
         close(20)
      enddo

      ! Convolve Green's functions with STF and accumulate weighted by solution
      ! This replicates the kernel construction logic in sub.r_green_H.f90
      do icm=1,icmn; do n=1,nn; do m=1,mn
               ! Convolution: green = gb * so
               ! Reference: sub.lapack_inv.f90::conv_y()
               call conv_y(gb(1:ndg,m,n,icm),So,green,iw)
               do jt=1,jtn(m,n)
                  l = l_id_m(m,n,jt,icm)
                  val = a(l)
                  if (abs(val) > 1.0e-20) then
                     ! Resample to 0.1s and apply time shift
                     ! Time shift = rupture_time + STF_delay + station_correction + GF_origin
                     call resample_shift(green,iw,dtg,Umn_s,ndjs,dts,Tr(m,n)+rtime*(jt-1)+cp(j)+tg0)
                     ! Accumulate: synthetic = sum of (solution * Green's function)
                     syn_s(1:ndjs) = syn_s(1:ndjs) + val * Umn_s(1:ndjs)
                  endif
               enddo
            enddo; enddo; enddo

      ! Apply taper to tail for consistency with inversion kernel
      ! Reference: sub.r_green_H.f90 Line 71, sub.c_wave_lib.f90::taper_tail()
      ! This ensures the 0.1s output matches the processing applied during inversion
      nwk = max(nint(ndjs*0.025),nint(2.0/dts))
      call taper_tail(syn_s, ndjs, nwk)

      ! Write synthetic waveform to file
      write(title, '(a, a, ".syn01")') trim(stcd(j)), trim(comp(j))
      open(82, file='wave.syn/'//trim(title))
      do i = 1, ndjs
         write(82, '(f12.4, 1x, e15.7)') (i-1)*dts, syn_s(i)
      end do
      close(82)

      !-------------------------------------------------------------------------
      ! PART 2b: High-Resolution Observed Waveform (dt = 0.1s)
      !-------------------------------------------------------------------------
      !
      ! Processing Logic (following sub.r_OBS.f90::readOBS_f):
      !   1. Read raw observation from wave.obs/ directory
      !   2. Apply offset correction using pre-P samples
      !      - Reference: sub.c_wave_lib.f90::offset()
      !   3. Resample to 0.1s with time shift = -p_sec
      !   4. Apply taper_tail for consistency with inversion data processing
      !
      ! File Format (wave.obs/):
      !   Header: Tp, dt, nt, lat, lon, tmax, nend, sigma
      !   Data:   waveform values
      !
      !-------------------------------------------------------------------------
      ic1=index(stcd(j),' ') - 1
      open(85, file='wave.obs/'//stcd(j)(1:ic1)//comp(j))
      read(85,*) t1_tmp, dt0_obs, nd0_obs, junk1, junk2, junk3, nend_obs, sig_obs
      nend_obs = nend_obs - nint(10./dt0_obs)
      np0_obs = nint(t1_tmp/dt0_obs) - 1
      allocate(wv_obs(nd0_obs), wvv_obs(ndjs))
      read(85,*) (wv_obs(i), i=1, nd0_obs)
      close(85)

      ! Remove offset using pre-P samples (same as readOBS_f)
      call offset(wv_obs, nd0_obs, np0_obs)
      ! Resample to 0.1s with time alignment to inversion window
      shift_obs = - p_sec
      call resample_shift(wv_obs, nd0_obs, dt0_obs, wvv_obs, ndjs, dts, shift_obs)

      ! Apply taper to tail for consistency with inversion data vector
      ! Reference: sub.r_OBS.f90::vector_d() applies taper_tail before assembly
      nwk = max(nint(ndjs*0.025),nint(2.0/dts))
      call taper_tail(wvv_obs, ndjs, nwk)

      ! Write observed waveform to file
      write(title, '(a, a, ".obs01")') trim(stcd(j)), trim(comp(j))
      open(83, file='wave.syn/'//trim(title))
      do i = 1, ndjs
         write(83, '(f12.4, 1x, e15.7)') (i-1)*dts, wvv_obs(i)
      end do
      close(83)
      deallocate(wv_obs, wvv_obs)

      deallocate(syn_s, Umn_s, green, so, gb)
   end do
   write(6, *) '0.1s Synthetic and Observed waveforms saved to wave.syn/*.syn01/*.obs01'
   !############################################################################
   d(1:ndata) = d(1:ndata) - zz1(1:ndata)    ! d(k) = d(k) - H(k,l)*a(l)
   wk2 = f_nrom2(ndata,d)
   !---
   var = wk2/wk1
   write(6,'("check: ",3(f11.4,1x),f9.5,1x,f13.1)') para1,para2,para3,var,s
   !---
   !     Out Put Final Solution
   ABIC  = 0.
   if(cmode.eq."F")then
      cp1 = 0.
      call writesol(mn,nn,jtn,jtn0,l_id_m,icmn,a,tr,var,abic,strike,dip,slip,    &
         m0,n0,xx,yy,rtime,beta,siga,depth,stcd,comp,sigw,dt,   &
         cp,jn,vr0,nsurface,rslip,ylength,alpha1,alpha2)
   endif
   deallocate(wobs,zz1)
   !---
   stop
END Program inversionA_TAd
