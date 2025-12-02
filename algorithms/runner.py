
def get_runner_cls(algorithm):
    if algorithm == "nfsp":
        from algorithms.nfsp.run_nfsp import RunNFSP

        return RunNFSP

    if algorithm == "escher":
        from algorithms.escher.run_escher_parallel import RunEscherParallel

        return RunEscherParallel

    if algorithm == "psro":
        from algorithms.psro.run_psro import RunPSRO

        return RunPSRO

    if algorithm == "rnad":
        from algorithms.rnad.run_rnad import RunRNaD

        return RunRNaD

    if algorithm == "ppo":
        from algorithms.ppo.run_ppo import RunPPO

        return RunPPO

    if algorithm == "fixppo":
        from algorithms.fixppo.run_fixppo import RunFIXPPO

        return RunFIXPPO
    
    if algorithm == "mmd":
        from algorithms.mmd.run_mmd import RunMMD

        return RunMMD

    if algorithm == "ppg":
        from algorithms.ppg.run_ppg import RunPPG

        return RunPPG

    raise ValueError(f'Unrecognized algorithm: {algorithm}')
