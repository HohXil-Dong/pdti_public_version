#!/bin/bash
str=182.
dip=15.
rake=70.
Rslip=0.
depth=21.
XX=5.
YY=5.
MN=15
NN=15
M0=8
N0=8
ICMN=5
ns=2
vr=2.5
TR=0.7
JTN=29
st_max=45
r_s_t=$vr
tl=70.
dt=$TR
dtg=0.1
dump=0.5
title=$(pwd)
precision=sp
export OMP_NUM_THREADS=30
#---------------------------------
#-----------------------------------------------------
#  Rotation Basis Double Couple component
#  Original value according to Kikuchi & Kanamori,(1991) : (strike, dip, rake)=(0., 90., 0.)
#  By properly rotating the basis tensor, the time-addaptive smoothing will work properly.
#  The best way to do this would be to put in the values determined by the GCMT.
#-----------
#02305050542A NEAR WEST COAST OF HONSH5
#  Date: 2023/ 5/ 5   Centroid Time:  5:42: 9.1 GMT
#  Lat=  37.57  Lon= 137.13
#  Depth= 12.0   Half duration= 3.4
#  Centroid time minus hypocenter time:  4.7
#  Moment Tensor: Expo=25  2.530 -1.320 -1.210 -1.270 -2.010 -1.390 
#  Mw = 6.3    mb = 0.0    Ms = 6.2   Scalar Moment = 3.52e+25
#  Fault plane:  strike=56    dip=25   slip=108
#  Fault plane:  strike=217    dip=66   slip=82
#-----------
rstike=182.
rdip=15.
rrake=70.
echo $rstike  $rdip   $rrake   >  RefMomentStrDipRake.dat
PDTI_fgenrotfivetensors
#-----------------------------------------------------
PDTI_TA_main.bash $str $dip $rake $Rslip $depth $vr $XX $YY $MN $NN $M0 $N0 $ICMN $TR $JTN $ns $dt $dtg $tl $dump $st_max $r_s_t $title $precision
#--  [  Rotate fort.40 ]
cp fort.40 fort.40back
PDTI_fgenrotfort40
rm -rf  RefMomentStrDipRake.dat  Rotation_basis_DC.dat
#-------------------------------------------------
PDTI_SaveInversionResults.bash
#-------------------------------------------------
