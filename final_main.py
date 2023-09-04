import os
import streamlit as st
# from receipt_analysis import receipt_ananlysis  # Import your receipt analysis function
from receipt import receipt_ananlysis
def main():
    st.title("Receipt Extraction")

    # Create "docs" directory if it doesn't exist
    DOCS_FOLDER = "receipt_storage"
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)

    # st.sidebar.title("Document Uploader")

    # File uploader in the sidebar
    # uploaded_files = st.sidebar.file_uploader("Upload multiple .jpg, .png, or .pdf files",type=["jpg", "png", "jpeg"], accept_multiple_files=False)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Save the uploaded image to the 'uploads' folder
        with open(os.path.join("receipt_storage", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Provide a link to download the uploaded image
        st.markdown(f"### [Download the image](uploads/{uploaded_file.name})")
        if st.button("Extract Text"):
            # Save the uploaded image to a temporary location
            with open("temp_image.jpg", "wb") as temp_image:
                temp_image.write(uploaded_file.read())

            # Call the receipt_analysis function to extract text
            out = receipt_ananlysis("temp_image.jpg")
            # json_output={}
            # for i in range(len(out['ExpenseDocuments'][0]["SummaryFields"])):
            #     complete_response=out['ExpenseDocuments'][0]["SummaryFields"][i]
            #     json_output[complete_response['Type']["Text"]]=complete_response['ValueDetection']["Text"]
            # for i in range(len(out['ExpenseDocuments'][0]["LineItemGroups"][0]["LineItems"][0]["LineItemExpenseFields"])):
            #     complete_response=out['ExpenseDocuments'][0]["LineItemGroups"][0]["LineItems"][0]["LineItemExpenseFields"][i]
            #     json_output[complete_response['Type']["Text"]]=complete_response['ValueDetection']["Text"]

            # Display the extracted text
            st.header("Extracted Text:")
            st.text(out)

            # Remove the temporary image
            os.remove("temp_image.jpg")


    # Section to analyze uploaded receipts
    # st.header("Analyze Receipts")

    # if uploaded_files:
    #     st.subheader("Uploaded Receipts")

    #     # Display uploaded receipt images
    #     for file in uploaded_files:
    #         st.image(file, caption=f"Uploaded Receipt: {file.name}", use_column_width=True)

    #     # Analyze uploaded receipts
    #     for file in uploaded_files:
    #         if file.type == "application/pdf":
    #             st.write(f"Analyzing PDF receipt: {file.name}")
    #             # Assuming you have a function to extract images from a PDF (not shown here)
    #             # Extract images from PDF and pass them to the receipt analysis function
    #             pdf_images = extract_images_from_pdf(file_path)
    #             for image in pdf_images:
    #                 analysis_result = receipt_ananlysis(image)
    #                 st.subheader(f"Analysis Result for {file.name}")
    #                 st.json(analysis_result)
    #         elif file.type in ["image/jpeg", "image/png"]:
    #             st.write(f"Analyzing image receipt: {file.name}")
    #             analysis_result = receipt_ananlysis(file_path)
    #             st.subheader(f"Analysis Result for {file.name}")
    #             st.json(analysis_result)
main()