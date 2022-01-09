import os
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
import torch
from torch.utils.data.dataloader import DataLoader

from builder import DocumentBuilder
from trocr import IAMDataset, device, get_processor_model
from doctr.utils.visualization import visualize_page
from doctr.models.predictor.base import _OCRPredictor
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.preprocessor import PreProcessor
from doctr.models import db_resnet50, db_mobilenet_v3_large

from doctr.io import DocumentFile
import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st

DET_ARCHS = ["db_resnet50", "db_mobilenet_v3_large"]
RECO_ARCHS = ["microsoft/trocr-large-printed", "microsoft/trocr-large-stage1", "microsoft/trocr-large-handwritten"]


def main():
    # Wide mode
    st.set_page_config(layout="wide")
    # Designing the interface
    st.title("docTR + TrOCR")
    # For newline
    st.write('\n')
    #
    st.write('For Detection DocTR: https://github.com/mindee/doctr')
    # For newline
    st.write('\n')
    st.write('For Recognition TrOCR: https://github.com/microsoft/unilm/tree/master/trocr')
    # For newline
    st.write('\n')
    
    st.write('Any Issue please dm: https://twitter.com/kforcode')
    # For newline
    st.write('\n')
    # Instructions
    st.markdown(
        "*Hint: click on the top-right corner of an image to enlarge it!*")
    # Set the columns
    cols = st.columns((1, 1, 1))
    cols[0].subheader("Input page")
    cols[1].subheader("Segmentation heatmap")
    
    # Sidebar
    # File selection
    st.sidebar.title("Document selection")
    # Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader(
        "Upload files", type=['pdf', 'png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.pdf'):
            doc = DocumentFile.from_pdf(uploaded_file.read()).as_images()
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        page_idx = st.sidebar.selectbox(
            "Page selection", [idx + 1 for idx in range(len(doc))]) - 1
        cols[0].image(doc[page_idx])
    # Model selection
    st.sidebar.title("Model selection")
    det_arch = st.sidebar.selectbox("Text detection model", DET_ARCHS)
    rec_arch = st.sidebar.selectbox("Text recognition model", RECO_ARCHS)
    # For newline
    st.sidebar.write('\n')
    if st.sidebar.button("Analyze page"):
        if uploaded_file is None:
            st.sidebar.write("Please upload a document")
        else:
            with st.spinner('Loading model...'):
                if det_arch == "db_resnet50":
                    det_model = db_resnet50(pretrained=True)
                else:
                    det_model = db_mobilenet_v3_large(pretrained=True)
                det_predictor = DetectionPredictor(PreProcessor((1024, 1024), batch_size=1, mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287)), det_model)
                rec_processor, rec_model = get_processor_model(rec_arch)
            with st.spinner('Analyzing...'):
                # Forward the image to the model
                processed_batches = det_predictor.pre_processor([doc[page_idx]])
                out = det_predictor.model(processed_batches[0], return_model_output=True)
                seg_map = out["out_map"]
                seg_map = torch.squeeze(seg_map[0, ...], axis=0)
                seg_map = cv2.resize(seg_map.detach().numpy(), (doc[page_idx].shape[1], doc[page_idx].shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
                # Plot the raw heatmap
                fig, ax = plt.subplots()
                ax.imshow(seg_map)
                ax.axis('off')
                cols[1].pyplot(fig)

                # Plot OCR output
                # Localize text elements
                loc_preds = out["preds"]

                # Check whether crop mode should be switched to channels first
                channels_last = len(doc) == 0 or isinstance(doc[0], np.ndarray)

                # Crop images
                crops, loc_preds = _OCRPredictor._prepare_crops(
                    doc, loc_preds, channels_last=channels_last, assume_straight_pages=True
                )

                test_dataset = IAMDataset(crops[0], rec_processor)
                test_dataloader = DataLoader(test_dataset, batch_size=16)

                text = []
                with torch.no_grad():
                    for batch in test_dataloader:
                        pixel_values = batch["pixel_values"].to(device)
                        generated_ids = rec_model.generate(pixel_values)
                        generated_text = rec_processor.batch_decode(
                            generated_ids, skip_special_tokens=True)
                        text.extend(generated_text)
                boxes, text_preds = _OCRPredictor._process_predictions(
                    loc_preds, text)

                doc_builder = DocumentBuilder()
                out = doc_builder(
                    boxes,
                    text_preds,
                    [
                        # type: ignore[misc]
                        page.shape[:2] if channels_last else page.shape[-2:]
                        for page in [doc[page_idx]]
                    ]
                )
                
                for df in out:
                    st.markdown("text")
                    st.write(" ".join(df["word"].to_list()))
                    st.write('\n')
                    st.markdown("\n Dataframe Output- similar to Tesseract:")
                    st.dataframe(df)
                    


if __name__ == '__main__':
    main()
