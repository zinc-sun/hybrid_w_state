# Clean W + GHZ rate notebooks

Put `atom_based.ipynb`, `hybrid.ipynb`, and the included `helper.py` in one
folder. Install NumPy, SciPy, SymPy, pandas, dill, and a Python Jupyter kernel.
Run each notebook top to bottom. The final cell runs the full distance sweep.

Use the helper supplied in this bundle: it also includes the rate and optimizer
functions, which are not present in the earlier merge-only helper. Its numerical
engine is extracted from the preceding overall-rate notebooks, so this is a
refactoring of those notebooks, not a new noise model or correction convention.

Set SOURCE_DIR in each notebook to your trusted source-expression folder:

    atom_based_prob_click.dill
    atom_based_ion_dm.dill
    hybrid_prob_click.dill
    hybrid_prob_load.dill
    hybrid_ion_dm.dill

These expressions are not bundled or changed. Use the corrected hybrid source
exports containing both QM-readout and QFC loss. There are no automatic source
hash checks, sanity-check cells, validation reports, or test scripts in this bundle.
Basic numerical error handling and zero-probability guards remain in the helper.

The notebooks return W at level 0 and W + GHZ at levels 1 and 2. All accepted
routes use one common clock. Optimization maximizes the total at one common q
(or q and lambda), with the same robust search algorithms and bounds as before.

POSTPROCESSING: one_way, ad (forced), or best (optional).
FINAL_POOLING: route or class. Fine measurement outcomes are corrected and averaged.

get_key_rate(...) returns total bits/s.
get_key_rate_components(...) returns the W/GHZ/total breakdown and route rates.
optimize_rate(...) returns the best evaluated parameters in .x and -rate in .fun.
run_overall_sweeps(...) returns tables, optimizer_results, route_rates.

Only the main rate CSVs are written. No separate diagnostic/manifest files are
written automatically. A CSV is checkpointed after each distance point.
keyrate_bps_level_N already denotes the total, not the W component.

Timing and source assumptions are unchanged: perfect memories, fixed-slot
hybrid attempts, 11/6 at the first level, and factor 3 at the second. These are
conservative timing benchmarks, not scheduling-optimal or finite-key rates.
