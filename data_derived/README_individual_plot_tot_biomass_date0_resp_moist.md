# README - individual_plot_tot_biomass_date0_resp_moist

## Source files

- `data/individual_plot_tot_biomass_flux.xlsx`
- `data/data_set_published.xlsx`, sheet `data_set_published`

## Output files

- `data_derived/individual_plot_tot_biomass_date0_resp_moist.xlsx`
- `data_derived/individual_plot_tot_biomass_date0_resp_moist_mean_over_plot.xlsx`
- `data_derived/individual_plot_tot_biomass_date0_resp_moist_mean_over_plot.xlsx`, sheet `individual_biomasses`
- `data_derived/individual_plot_tot_biomass_date0_resp_moist_mean_over_plot.xlsx`, sheet `rates`

## Processing steps

1. Loaded `individual_plot_tot_biomass_flux.xlsx`.
2. Removed the columns `tot_flux` and `dominant_real_eigenvalue`.
3. Reconstructed the following columns from `Unique ID`:
   - `Location`
   - `Site`
   - `Management`
   - `Treatment`
   - `Plot`
   - `Date`
4. Converted `Date` to a numeric value by removing the `d` prefix:
   - example: `d0` -> `0`
5. Filtered the dataset to keep only rows with `Date = 0`.
6. Loaded the columns `id`, `Resp`, and `Moist` from `data_set_published.xlsx`, sheet `data_set_published`.
7. Merged the datasets using:
   - `Unique ID` from the first dataset
   - `id` from the published dataset
8. Added `Resp` and `Moist` to the final dataset.
9. Created a second derived dataset by averaging across `Plot` (`p1`, `p2`, `p3`), while keeping `Location`, `Site`, `Management`, `Treatment`, and `Date` fixed.
10. For this grouped dataset:
   - all numeric variables were averaged across the available rows in each group
   - missing values were ignored when calculating the mean
   - if a variable was missing for all rows in a group, the output remains `NaN`
   - identifier columns that cannot be averaged, such as `Unique ID`, `Plot`, and `Unnamed: 0`, were not included in the grouped output
11. Added a new sheet named `individual_biomasses` to `individual_plot_tot_biomass_date0_resp_moist_mean_over_plot.xlsx`.
12. Loaded the 10 individual-biomass columns from `FoodWeb_vectorsFinal.xlsx`, using the sheet `FoodWebVectors`:
   - `predmite_M`
   - `prednem_M`
   - `coll_M`
   - `orib_M`
   - `detmite_M`
   - `plantfeed_M`
   - `ppnnem_M`
   - `omninem_M`
   - `fungnem_M`
   - `bactnem_M`
13. Filled the `individual_biomasses` sheet by matching rows on:
   - `Location`
   - `Site`
   - `Management`
   - `Treatment`
14. Kept the same row order as the main plot-averaged sheet and added a `row_index` column so the rows can be referenced consistently in code.
15. Added a new sheet named `rates` to `individual_plot_tot_biomass_date0_resp_moist_mean_over_plot.xlsx`.
16. Loaded the rate columns ending with `_met` from `FoodWeb_vectorsFinal.xlsx`, using the sheet `Rates`.
17. Filled the `rates` sheet by matching rows on:
   - `Location`
   - `Site`
   - `Management`
   - `Treatment`
18. Kept the same row order as the main plot-averaged sheet and added a `row_index` column so the rows can be referenced consistently in code.

## Notes

- The final dataset contains `178` rows.
- The plot-averaged dataset contains `60` rows.
- The `Unnamed: 0` column from the original file was kept because it was not included among the columns to remove.
- For `7` records, no match was found in the `data_set_published` sheet; in those cases `Resp` and `Moist` remain empty.
- The plot-averaged dataset groups rows by `Location`, `Site`, `Management`, `Treatment`, and `Date`, then averages across the available plot replicates.
- Two groups contain only `2` plot rows instead of `3`; these groups were still retained and averaged over the available plots only.
- The workbook now contains three sheets: `Sheet1`, `individual_biomasses`, and `rates`.
- The `individual_biomasses` sheet has the same 60-row order as `Sheet1`.
- The `rates` sheet has the same 60-row order as `Sheet1`.
- One rate value is missing in the source `Rates` sheet: `plantfeed_met` for `Sc / S1 / In / D` is already `NaN` in the original file and remains `NaN` in the derived sheet.

## IDs without a match for `Resp` and `Moist`

- `DeS2InCp3d0`
- `ScS1InDp1d0`
- `ScS1InDp2d0`
- `ScS1InDp3d0`
- `ScS3InDp1d0`
- `YoS2InCp3d0`
- `YoS5ExDp2d0`
