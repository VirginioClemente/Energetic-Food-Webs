# README - Dataset_analysis_CV

## Source files

- `data/individual_plot_tot_biomass_flux.xlsx`
- `data/FoodWeb_vectorsFinal.xlsx`, sheet `FoodWebVectors`

## Output file

- `data_derived/Dataset_analysis_CV.xlsx`
- `data_derived/Dataset_analysis_CV.xlsx`, sheet `rates`

## Processing steps

1. Loaded `individual_plot_tot_biomass_flux.xlsx`.
2. Parsed `Unique ID` into:
   - `Location`
   - `Site`
   - `Management`
   - `Treatment`
   - `Plot`
   - `Date`
3. Converted `Date` from the original code format to a numeric value:
   - example: `d20` -> `20`
4. Removed the columns `tot_flux` and `dominant_real_eigenvalue`, following the same cleanup used for the previous derived dataset.
5. Filtered the dataset to keep only rows with `Treatment = C`.
6. Loaded the 10 individual-biomass columns from `FoodWeb_vectorsFinal.xlsx`, sheet `FoodWebVectors`:
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
7. Merged those `*_M` columns into the filtered dataset using:
   - `Location`
   - `Site`
   - `Management`
   - `Treatment`
8. Computed `SumN` using only the first 10 consumer biomass groups:
   - for each group, calculated `*_B / *_M`
   - summed those 10 values row by row
   - ignored missing values in the sum
   - kept `NaN` only if all 10 components were missing for a row
9. Added a new sheet named `rates` to `Dataset_analysis_CV.xlsx`.
10. Loaded the rate columns ending with `_met` from `FoodWeb_vectorsFinal.xlsx`, using the sheet `Rates`.
11. Filled the `rates` sheet by matching rows on:
   - `Location`
   - `Site`
   - `Management`
   - `Treatment`
12. Kept the same row order as the main sheet and added a `row_index` column so the rows can be referenced consistently in code.

## Notes

- The final dataset contains `351` rows.
- Only `Treatment = C` rows are included.
- The `Date` values present in the dataset are `0`, `8`, `20`, and `60`.
- `SumN` uses only the first 10 consumer groups and does not include the basal resources:
   - `fungi_B`
   - `bact_B`
   - `roots_B`
- All rows successfully received the 10 `*_M` columns after the merge.
- The workbook now contains two sheets: `Sheet1` and `rates`.
- The `rates` sheet has the same 351-row order as `Sheet1`.
