## HW01: Neuronetwork Trained with XOR Data
Yueh-Shun Li

### Build instructions

#### How to build
```bash
make
```

#### How to run
The proof-of-concept main function is src/main/nnmicro-xor-demo.c. Build it with `make` and run with `./build/nnmicro-xor-demo`.


#### How to run test cases
```
make check-(arr|idx|libnn|plain_allocation) # Execute each test cases
make check_valgrind-(arr|idx|libnn|plain_allocation) # Execute with valgrind
make check-run # Execute all test cases
make check-valgrind # Execute all with valgrind
make check # Do the above two
```

### Design
The code implements the hand-calculated gradient-descending algorithm, and terminates after either
1.  it converges to a low-enough loss specified by `max_loss` and `max_delta_loss`, or
2.  the loss refuses to go down after a given (10) times specified by `max_retrial`

The project hierarchy is:
*   `include/*.h` for header files
*   `src/main/*.c` for C files containing main functions
*   `src/*.c` for library implementations
*   `external_projects/[Author]/[Project Name]/` for the repository of third-party open-source projects.
*    The project root for `.git*`, `Makefile`, `README.md`, etc.

The project structure is planned to be as general and reusable as possible. However, the time to write and debug such a project turns out to be unaffordable. The main function is thus reduced into a two-input-node, one-hidden-layer 200-liner with parameters hard-coded (`src/nnmicro-xor-demo.c`)


### Analysis
The loss keeps "converging" toward 0.25 and the output to 0.5. This tendency appears across different initial weights. The reason is still under investigation.

Sample output (see `result.log` for full output):

Line 1
```=1
Input		Answer
(0.000000, 0.000000)	0.000000
(1.000000, 0.000000)	1.000000
(0.000000, 1.000000)	1.000000
(1.000000, 1.000000)	0.000000
loss_now, loss_delta_neg: 0.297332, 1.79769e+308
loss_now, loss_delta_neg: 0.305528, -0.00819601
loss_now, loss_delta_neg: 0.305705, -0.000176703
loss_now, loss_delta_neg: 0.305677, 2.78774e-05
loss_now, loss_delta_neg: 0.305634, 4.2948e-05
loss_now, loss_delta_neg: 0.30558, 5.37032e-05
loss_now, loss_delta_neg: 0.305516, 6.43491e-05
loss_now, loss_delta_neg: 0.305441, 7.49788e-05
loss_now, loss_delta_neg: 0.305355, 8.55921e-05
loss_now, loss_delta_neg: 0.305259, 9.61867e-05
loss_now, loss_delta_neg: 0.305153, 0.00010676
loss_now, loss_delta_neg: 0.305035, 0.00011731
loss_now, loss_delta_neg: 0.304907, 0.000127834
loss_now, loss_delta_neg: 0.304769, 0.00013833
loss_now, loss_delta_neg: 0.30462, 0.000148795
```

Line 76
```=76
loss_now, loss_delta_neg: 0.281913, 0.000587871
loss_now, loss_delta_neg: 0.281322, 0.000590925
loss_now, loss_delta_neg: 0.280729, 0.000593724
loss_now, loss_delta_neg: 0.280132, 0.000596264
loss_now, loss_delta_neg: 0.279534, 0.000598539
loss_now, loss_delta_neg: 0.278933, 0.000600547
loss_now, loss_delta_neg: 0.278331, 0.000602282
loss_now, loss_delta_neg: 0.277727, 0.000603739
loss_now, loss_delta_neg: 0.277122, 0.000604916
loss_now, loss_delta_neg: 0.276517, 0.000605808
loss_now, loss_delta_neg: 0.27591, 0.00060641
loss_now, loss_delta_neg: 0.275303, 0.00060672
loss_now, loss_delta_neg: 0.274697, 0.000606733
loss_now, loss_delta_neg: 0.27409, 0.000606445
loss_now, loss_delta_neg: 0.273484, 0.000605854
loss_now, loss_delta_neg: 0.272879, 0.000604956
loss_now, loss_delta_neg: 0.272276, 0.000603748
loss_now, loss_delta_neg: 0.271674, 0.000602227
loss_now, loss_delta_neg: 0.271073, 0.00060039
loss_now, loss_delta_neg: 0.270475, 0.000598234
```

Line 141
```=141
loss_now, loss_delta_neg: 0.25081, 0.000165162
loss_now, loss_delta_neg: 0.250659, 0.000150112
loss_now, loss_delta_neg: 0.250524, 0.000134952
loss_now, loss_delta_neg: 0.250405, 0.000119691
loss_now, loss_delta_neg: 0.2503, 0.00010434
loss_now, loss_delta_neg: 0.250212, 8.89123e-05
loss_now, loss_delta_neg: 0.250138, 7.34176e-05
loss_now, loss_delta_neg: 0.25008, 5.7868e-05
loss_now, loss_delta_neg: 0.250038, 4.22751e-05
loss_now, loss_delta_neg: 0.250011, 2.66504e-05
loss_now, loss_delta_neg: 0.25, 1.10058e-05
loss_now, loss_delta_neg: 0.250005, -4.64712e-06
loss_now, loss_delta_neg: 0.250025, -2.02965e-05
loss_now, loss_delta_neg: 0.250061, -3.59307e-05
loss_now, loss_delta_neg: 0.250113, -5.15379e-05
loss_now, loss_delta_neg: 0.25018, -6.71066e-05
loss_now, loss_delta_neg: 0.250262, -8.26249e-05
loss_now, loss_delta_neg: 0.250361, -9.80815e-05
loss_now, loss_delta_neg: 0.250474, -0.000113465
loss_now, loss_delta_neg: 0.250603, -0.000128763
loss_now, loss_delta_neg: 0.250747, -0.000143966
Input	Raw output	Output	Answer
(0, 0)	0.530097	1	0
(1, 0)	0.530097	1	1
(0, 1)	0.530097	1	1
(1, 1)	0.530097	1	0
```

### Reference

#### Dependent software projects

*   Danial Abrecht (@Daniel-Abrecht). IEEE754 binary encoder: A C library for converting float and double values to binary. https://github.com/Daniel-Abrecht/IEEE754_binary_encoder
*   Christopher Wellons (@skeeto) et. al.. optparse: Portable, reentrant, getopt-like option parser. https://github.com/skeeto/optparse

#### Makefile and project hierarchy

*   Hilton Lipschitz (@hiltmon). A Simple C++ Project Structure. https://hiltmon.com/blog/2013/07/03/a-simple-c-plus-plus-project-structure/

#### Building and Testing infrastructure

*   GNU. GCC, the GNU Compiler Collection. https://gcc.gnu.org/
*   GNU. GDB: The GNU Debugger. https://www.sourceware.org/gdb/
*   Valgrind Developers. Valgrind. https://valgrind.org/

#### Suggestions and answers

*   GNU. The GNU C Reference Manual. https://www.gnu.org/software/gnu-c-manual/gnu-c-manual.html
*   Various StackOverflow posts, some documented as comment inside this project.
