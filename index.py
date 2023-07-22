# %% [markdown]
# ChromatinHD analyzes single-cell ATAC+RNA data using the raw fragments as input, by automatically adapting the scale at which
# relevant chromatin changes on a per-position, per-cell, and per-gene basis.
#
# Currently, the following models are supported:
#
# <ul>
#     <li><strong><i>pred</i></strong>: Predicting gene expression from fragments
#         <ul>
#             <li>To learn where accessibility is predictive for gene
#                 expression</li>
#             <li>To learn which regions in the genome are likely collaborating to regulate gene expression</li>
#             <li>To learn for which regions high fragment sizes are indicative of gene expression, which indicates an active regulatory region with dense protein binding</li>
#         </ul>
#     </li>
#     <li><strong><i>diff</i></strong>: Understanding the differences in accessibilty between cell types/states</li>
# </ul>
