# TLBfind

TLBfind is a open-source software for simulations of
concentrated emulsions with finite-size droplet under
thermal convection. This code lays foundations on
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

TLBfind is supplied as a single directory. Here the user can find:
- the preprint version of the article (`TLBfind_paper.pdf`, https://arxiv.org/pdf/2109.12565)
- the source code file `cudatlbfind.cu`
- `tlbfind_testCase*.inp` input files, containing different test cases explained in the
  article that can help the user to edit new input files (see `README_hoWTo*` files for
  details)
- analysis programs: `*.cu` and `*.c` (see `README_analysis` file for details)
- gnuplot scripts to reproduce figures in the paper (see `README_hoWTo*` files for
  details)

### Compiling and executing TLBfind

To run and compile TLBfind, follow the instructions in the `README_howToPreparation`
file.
