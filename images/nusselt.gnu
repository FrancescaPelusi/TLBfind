plotTitle = sprintf("nusselt")
texOut = sprintf("%s.tex", plotTitle)

fontSize = 20.5

size_key = ',22'
size_label = ',28'
size_tics = ',30'

LW = 7
set term epslatex 'default' color dashed colortext standalone font fontSize
set out texOut

x_main_size = 2.2
y_main_size = 1.4

set size x_main_size,y_main_size

set key top right box width 4. font size_key
set key spacing 1.6
unset key

set lmargin screen 0.3

set xlabel 'simulation time' font ',1'
set ylabel 'Nu' font size_label

set xtics('$2 \ 10^{5}$' 200000,'$4 \ 10^{5}$' 400000,'$6 \ 10^{5}$' 600000,'$8 \ 10^{5}$' 800000,'$10^{6}$' 1000000)
set ytics('$2.9$' 2.9,'$2.6$' 2.6,'$2.7$' 2.7,'$2.8$' 2.8,'$3.0$' 3.,'$3.1$' 3.1,'$3.2$' 3.2,'$3.3$' 3.3,'$3.4$' 3.4)

f(x) = 2.83

p[2e5:1e6][2.55: 3.1] f(x) w l lw 8. dt 3 lc rgb 'brown' t '' ,\
'timeNusseltNumber.dat' u 1:($2) every 2 w p pt 7 ps 4. lc rgb 'black' t '' ,\
'timeNusseltNumber.dat' u 1:($2) every 2 w p pt 7 ps 3. lc rgb '#008b8b' t ''

set out

cmd = sprintf("latex %s; dvipdf %s.dvi; dvips %s.dvi; ps2eps %s.ps", texOut, plotTitle, plotTitle, plotTitle)
system(cmd)
cmd = sprintf("rm %s.dvi %s.log %s.aux %s.tex", \
    plotTitle, plotTitle, plotTitle, plotTitle, plotTitle)
system(cmd)


