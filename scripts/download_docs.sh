#!/bin/bash
# download_docs.sh
# Downloads all pending knowledge-base PDFs into docs/
# Usage: bash scripts/download_docs.sh

set -e
DOCS="$(dirname "$0")/../docs"
cd "$DOCS"

echo "Downloading knowledge-base PDFs into docs/ ..."

# Tomato Early Blight (already downloaded by curl, skip if exists)
[ -f tomato_late_blight_wisc.pdf ]        || curl -L -o tomato_late_blight_wisc.pdf        "https://barron.extension.wisc.edu/files/2023/02/Tomato-Late-Blight.pdf"
[ -f tomato_early_blight_ctahr.pdf ]      || curl -L -o tomato_early_blight_ctahr.pdf      "https://www.ctahr.hawaii.edu/oc/freepubs/pdf/PD-45.pdf"
[ -f tomato_early_blight_vegres.pdf ]     || curl -L -o tomato_early_blight_vegres.pdf     "https://www.maxapress.com/data/article/vegres/preview/pdf/vegres-0025-0010.pdf"
[ -f tomato_nhb_guide.pdf ]               || curl -L -o tomato_nhb_guide.pdf               "https://nhb.gov.in/pdf/vegetable/tomato/tom002.pdf"

# Tomato Late Blight
[ -f tomato_early_blight_fungicide_mgmt.pdf ] || curl -L -o tomato_early_blight_fungicide_mgmt.pdf \
  "https://ijhssm.org/issue_dcp/Effective%20Management%20of%20Tomato%20Early%20Blight%20Alternaria%20solani%20Using%20Fungicide%20Applications.pdf"
[ -f tomato_late_blight_ijpab.pdf ]       || curl -L -o tomato_late_blight_ijpab.pdf       "https://www.ijpab.com/form/2019%20Volume%207,%20issue%202/IJPAB-2019-7-2-629-635.pdf"

# Tomato Leaf Mold
[ -f tomato_leaf_mold_ppqs.pdf ]          || curl -L -o tomato_leaf_mold_ppqs.pdf          "https://ppqs.gov.in/sites/default/files/tomato.pdf"
[ -f tomato_leaf_mold_lsu.pdf ]           || curl -L -o tomato_leaf_mold_lsu.pdf           "https://www.lsuagcenter.com/~/media/system/6/2/8/f/628fe95b838faf1eecf7ede119a486a7/pub3455tomatoleafmold.pdf"

# Late Blight / Potato
[ -f late_blight_disease_ppqs.pdf ]       || curl -L -o late_blight_disease_ppqs.pdf       "https://ppqs.gov.in/sites/default/files/late_blight_disease.pdf"
[ -f potato_nhb_guide.pdf ]               || curl -L -o potato_nhb_guide.pdf               "https://nhb.gov.in/pdf/vegetable/potato/pot002.pdf"
[ -f potato_blight_sensors_ncbi.pdf ]     || curl -L -o potato_blight_sensors_ncbi.pdf     "https://pmc.ncbi.nlm.nih.gov/articles/PMC11644959/pdf/sensors-24-07864.pdf"

# Corn Common Rust
[ -f corn_common_rust_tda.pdf ]           || curl -L -o corn_common_rust_tda.pdf           "https://www.corn-states.com/app/uploads/2018/07/Common20Rust20of20Corn-TDA.pdf"

echo "Done. PDFs in docs/:"
ls -lh *.pdf
