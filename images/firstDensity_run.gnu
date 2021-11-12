#reset

mult=2.
PX=1024*mult

sizeXR = 1.
sizeYR = 0.4

set terminal pngcairo size int(sizeXR*PX),int(sizeYR*PX) enhanced
outfile = sprintf('firstDensity_run.png')
set out outfile
unset tics
unset border
unset key

set pm3d map
set size ratio sizeYR/sizeXR

set lmargin at screen 0.
set rmargin at screen 1.
set tmargin at screen 1.
set bmargin at screen 0.

set multiplot
set palette rgb 33,13,10
unset colorbox
filename = sprintf("temperature.350.dat")
filename2 = sprintf("firstdensity.350.dat")
displacement = sprintf("./displacementAnalysis/deltaField")
sp[:947][:431][]filename u 1:2:(($3))

MAXC=1.4
MINC=0.1

set style arrow 2 size screen 0.025,15,45 ls 1 lc rgb 'black' lw 3.5 head filled

a=5.

set palette defined (-5 "#DAA520",MINC "black",4.*MAXC/6 "black", 6.9*MAXC/8 "black",(MAXC-MINC) "black")
sp [][][:1.1] filename2 u 1:2:($3), displacement u ($7 > 10 & $7 < 925 ? $7 : 1/0):8:(0.):($9*a):($10*a):(0.) w vector arrowstyle 2
unset multiplot
set out



