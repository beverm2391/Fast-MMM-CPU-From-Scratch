# Fast Multidimensional Matrix Multiplication on CPU from Scratch

This is my repo for following along to [Simon Boehm's](https://siboehm.com/about/) article, [Fast Multidimensional Matrix Multiplication on CPU from Scratch](https://siboehm.com/articles/22/Fast-MMM-on-CPU).

### Compiler Optimizations
- Using `-O3` for [optimization](https://clang.llvm.org/docs/CommandGuide/clang.html#cmdoption-O0)
- Generating arch or march specific code with `-arch <architecture>` or `-march=<cpu>` for [architecture](https://clang.llvm.org/docs/CommandGuide/clang.html#cmdoption-march) and [microarchitecture](https://clang.llvm.org/docs/CommandGuide/clang.html#cmdoption-mcpu) respectively.
- Using `-ffast-math` to allow the compiler to do associative floor math and promising that there will be no NaNs or Infs.