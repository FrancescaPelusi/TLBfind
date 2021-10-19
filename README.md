# TLBfind

TLBfind is a open-source software for simulations of
concentrated emulsions with finite-size droplet under
thermal convection. This code lays his foundations on
a multi-component Shan-Chen lattice Boltzmann model.

The code has been written by

Francesca Pelusi, FZJ-HI ERN, Erlangen, Germany  
Matteo Lulli, SUSTech, Guangdong, China  
Mauro Sbagaglia, ToV, Rome, Italy  
Massimo Bernaschi, IAC-CNR, Rome, Italy  

with the support of

Andrea Scagliarini, IAC-CNR, Rome, Italy  
Roberto Benzi, ToV, Rome, Italy  
Fabio Bonaccorso, IIT-CLNS, Rome, Italy

The code is licensed under the MIT License.

### Structure

TLBfind is supplied as a main UNIX directory. Here the user can find:
- the source code file cudatlbfind.cu
- the 'tlbfind_testCase*' input files, containing different test cases that can
  help the user to edit new input files (see README_hoWTo* file for further details)
- analysis programs (see README_hoWToRun file for further details)
- the preprint version of the article (https://arxiv.org/pdf/2109.12565)
- gnuplot scripts to reproduce figures in the paper (see README_hoWTo* file for
  further details)

### Compiling and executing TLBfind

To run and compile TLBfind, follow the instructions in the README_howToPreparation
files.
