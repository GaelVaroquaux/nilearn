"""
Comparison of Dictionary Learning and ICA for doing group analysis of
resting-state fMRI
=====================================================

An example applying dictionary learning and ICA to resting-state data,
visualizing resulting components using atlas plotting tools.

Dictionary learning is a sparsity based decomposition method for extracting
spatial maps. It extracts maps that are naturally sparse and usually cleaner
than ICA

    * Gael Varoquaux et al.
    Multi-subject dictionary learning to segment an atlas of brain spontaneous
    activity
    Information Processing in Medical Imaging, 2011, pp. 562-573, Lecture Notes
    in Computer Science

Available on hal
"""
import matplotlib
matplotlib.use('Qt4Agg')
### Load ADHD rest dataset ####################################################
from nilearn import datasets

adhd_dataset = datasets.fetch_adhd()
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

### Apply Decomposition estimators#############################################
from nilearn.decomposition import DictLearning, CanICA

n_components = 20

### Dictionary learning #######################################################
dict_learning = DictLearning(n_components=n_components, smoothing_fwhm=6.,
                             memory="nilearn_cache", memory_level=2,
                             verbose=1,
                             alpha=2,
                             random_state=0,
                             n_epochs=1)
### CanICA ####################################################################
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache",  memory_level=2,
                threshold=3.,
                verbose=1)

### Fitting both estimators ###################################################
estimators = [dict_learning, canica]
components_imgs = []

for estimator in estimators:
    print('[Example] Learning maps using %s model' % type(estimator).__name__)
    estimator.fit(func_filenames)
    print('[Example] Saving results')
    # Decomposition estimator embeds their own masker
    masker = estimator.masker_
    components_img = masker.inverse_transform(estimator.components_)
    components_img.to_filename('%s_resting_state.nii.gz' %
                               type(estimator).__name__)
    components_imgs.append(components_img)

### Visualize the results #####################################################
import matplotlib.pyplot as plt
from nilearn.plotting import plot_prob_atlas, find_xyz_cut_coords
from nilearn.image import index_img

print('[Example] Displaying')

# We select pertinent cut coordinates for displaying
cut_coords = find_xyz_cut_coords(index_img(components_imgs[0], 1))
for estimator, atlas in zip(estimators, components_imgs):
    fig = plt.figure()
    plot_prob_atlas(atlas, view_type="filled_contours",
                    title="%s" % estimator.__class__.__name__,
                    figure=fig,
                    cut_coords=cut_coords, colorbar=False)
plt.show()
