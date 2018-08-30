import dipy
from dipy.data import read_stanford_labels
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy
from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.streamline import save_trk
from dipy.tracking.streamline import Streamlines
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

tensor_model = dti.TensorModel(gtab)
tenfit = tensor_model.fit(data, mask=white_matter)

FA = fractional_anisotropy(tenfit.evals)
classifier = ThresholdTissueClassifier(FA, .2)
detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                             max_angle=30.,
                                                             sphere=default_sphere)
streamlines = LocalTracking(detmax_dg, classifier, seeds, affine, step_size=.5)

save_trk("deterministic_maximum_shm_coeff.trk", streamlines, affine,
         labels.shape)
print("----%s seconds -----" %(time.time()-start_time))
##execution time 17.1111240387 seconds##