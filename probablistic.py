from dipy.data import read_stanford_labels
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
from dipy.tracking.streamline import Streamlines
from dipy.reconst.shm import CsaOdfModel
from dipy.direction import ProbabilisticDirectionGetter
from dipy.data import small_sphere
from dipy.io.streamline import save_trk
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
import time


start_time = time.time()
hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.affine

seed_mask = labels == 2
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)

csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
csd_fit = csd_model.fit(data, mask=white_matter)
csa_model = CsaOdfModel(gtab, sh_order=6)
gfa = csa_model.fit(data, mask=white_matter).gfa
classifier = ThresholdTissueClassifier(gfa, .25)
fod = csd_fit.odf(small_sphere)
pmf = fod.clip(min=0)
prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                sphere=small_sphere)
streamlines_generator = LocalTracking(prob_dg, classifier, seeds, affine, step_size=.5)
save_trk("probabilistic_small_sphere.trk", streamlines_generator, affine, labels.shape)
prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                    max_angle=30.,
                                                    sphere=default_sphere)
streamlines_generator = LocalTracking(prob_dg, classifier, seeds, affine, step_size=.5)
save_trk("probabilistic_shm_coeff.trk", streamlines_generator, affine, labels.shape)
print("------%s seconds -----" % (time.time() - start_time))
