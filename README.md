# Gomoku MCTS
Parallel MCTS for Gomoku - IUB Parallel Programming project.

## Run
    python play.py
    python benchmark.py
    python test_gomoku.py

    mpiexec -n 4 python mpi_play.py
    mpiexec -n 4 python mpi_benchmark.py

    python -c "import os, platform; print(os.cpu_count(), platform.processor())"
    12 Intel64 Family 6 Model 154 Stepping 4, GenuineIntel
