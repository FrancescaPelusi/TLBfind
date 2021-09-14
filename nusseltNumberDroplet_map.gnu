#reset

mult=2.
PX=1024*mult

sizeXR = 1.
sizeYR = 0.4

set terminal pngcairo size int(sizeXR*PX),int(sizeYR*PX) enhanced
outfile = sprintf('nusseltNumberDroplet_map.png')
set out outfile
unset tics
unset border
unset key

set pm3d map
set size ratio sizeYR/sizeXR

set rmargin at screen 1.05
set tmargin at screen 0.95
set bmargin at screen 0.05

set multiplot
set colorbox user origin graph -0.055,graph -0.03 size graph 0.03,graph 1.05
set palette model RGB defined (0. "lemonchiffon", 12. "orangered4" )

filename = sprintf("./nusseltDroplets/timeNusseltNumberDroplets")
sp[:947][:431][]filename u ($0>0)?($3):(1/0):4:5 w p pt 7 palette ps 6.5

unset multiplot
set out





