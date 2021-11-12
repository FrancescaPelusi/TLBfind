#reset

mult=2.
PX=1024*mult

sizeXR = 1.
sizeYR = 0.4

set terminal pngcairo size int(sizeXR*PX),int(sizeYR*PX) enhanced
outfile = sprintf('firstDensity_preparation_final.png')
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

unset colorbox

filename = sprintf("firstdensity.dat")

MAXC=1.4
MINC=0.1

set palette defined (MINC "#0080ff", 2.2*MAXC/6 "black",7.4*MAXC/8 "black", (MAXC-MINC) "#DAA520", MAXC "#DAA520")
sp [:947][:431][:] filename u 1:2:(MAXC-$3)
unset multiplot
set out

print(outfile)
