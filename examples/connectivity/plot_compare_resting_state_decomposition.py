"""
Dictionary Learning and ICA for doing group analysis of resting-state fMRI
==========================================================================

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

Available on https://hal.inria.fr/inria-00588898/en/
"""
### Load ADHD rest dataset ####################################################
from nilearn import datasets

adhd_dataset = datasets.fetch_adhd()
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

#################################s#############################################
# Apply Decomposition estimator
from nilearn.decomposition import DictLearning, CanICA

n_components = 40

###############################################################################
# Dictionary learning
dict_learning = DictLearning(n_components=n_components,
                             memory="nilearn_cache", memory_level=2,
                             verbose=1,
                             alpha=7,
                             random_state=0,
                             n_epochs=1)
###############################################################################
# CanICA
canica = CanICA(n_components=n_components,
                memory="nilearn_cache", memory_level=2,
                threshold=3.,
                n_init=1,
                verbose=1)
# ###############################################################################
# # Fitting both estimators
estimators = [dict_learning, canica]
names = {dict_learning: 'Dictionary learning', canica: 'CanICA'}
components_imgs = []

for estimator in estimators:
    print('[Example] Learning maps using %s model' % names[estimator])
    estimator.fit(func_filenames)
    print('[Example] Saving results')
    # Decomposition estimator embeds their own masker
    masker = estimator.masker_
    components_img = masker.inverse_transform(estimator.components_)
    components_img.to_filename('%s_resting_state.nii.gz' %
                               names[estimator])
    components_imgs.append(components_img)


###############################################################################
# Visualize the results
from nilearn.plotting import plot_prob_atlas, find_xyz_cut_coords, show, \
    plot_stat_map
from nilearn.image import index_img

print('[Example] Displaying')

# We select relevant cut coordinates for displaying
indices = {dict_learning: 19, canica: 4}
cut_coords = find_xyz_cut_coords(index_img(components_imgs[0],
                                           indices[dict_learning]))
for estimator, atlas in zip(estimators, components_imgs):
    plot_prob_atlas(atlas, view_type="filled_contours",
                    title="%s" % names[estimator],
                    cut_coords=cut_coords, colorbar=False)
    plot_stat_map(index_img(atlas, indices[estimator]),
                  title="%s" % names[estimator],
                  cut_coords=cut_coords, colorbar=False)
show()
