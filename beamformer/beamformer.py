from beamformer.delay_and_sum import DelayAndSumBeamformer
from beamformer.fourier import FourierBeamformer
from beamformer.capon import CaponBeamformer
from beamformer.music import MusicBeamformer

default_configuration = {"fourier": {"compute_method": "gpu"},
                             "capon": {"compute_method": "cpu"},
                             "music": {"compute_method": "cpu"},
                         "delay_and_sum": {"compute_method": "cpu"}}
def generate_beamformer(beamformer_type: str, config: dict = None):
    if config is None:
        compute_method = default_configuration[beamformer_type]["compute_method"]
    else:
        compute_method = config[beamformer_type]["compute_method"]

    if beamformer_type == "fourier":
        return FourierBeamformer(type=compute_method)
    elif beamformer_type == "capon":
        return CaponBeamformer(type=compute_method)
    elif beamformer_type == "music":
        return MusicBeamformer(type=compute_method)
    elif beamformer_type == "delay_and_sum":
        return DelayAndSumBeamformer(type=compute_method)
    else:
        raise ValueError("Wrong beamformer type.")