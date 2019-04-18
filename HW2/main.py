from sktdl_tinyimagenet import experiment


@experiment.ex.automain
def main():
    experiment.train()
