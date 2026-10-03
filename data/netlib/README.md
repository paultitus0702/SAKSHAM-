# Netlib Benchmark Inputs

This directory is the default location for user-supplied, decompressed Netlib MPS files. The configured Phase 11 candidates are listed in [`benchmarks.json`](benchmarks.json): `afiro.mps`, `adlittle.mps`, and `blend.mps`.

Obtain the authentic compact-encoded instances from the [official Netlib LP data index](https://www.netlib.org/lp/data/) and place the original files here. The parser validates the Netlib stream checksums and expands the supported encoding before standard MPS parsing. SAKSHAM does not download benchmark data or include these files in the repository.

Run a configured instance with `NetlibRunner().run_benchmark("afiro")`, or pass another input directory using `NetlibRunner(benchmark_directory=...)`. The runner reports absent configured files as `INPUT_NOT_FOUND`. Local test models must be labeled `SYNTHETIC_TEST_FIXTURE`; filenames alone do not verify provenance.

The parser also accepts ordinary supported fixed/free MPS text. It does not support `RANGES`, `OBJSENSE`, Netlib mystery records, or multiple compact-encoded problems in one file; unsupported input is reported per benchmark rather than treated as successful validation.
