from streamlit import cli as stcli
import streamlit
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
from PIL import Image
import io
import sys

def main():
    streamlit.title('Enhanced Super Resolution GAN')

    # fastapi endpoint
    url = 'http://127.0.0.1:8000'
    endpoint = '/uploadfile/'
    col_1, col_2 = streamlit.beta_columns(2)
    col_1.image("https://pytorch.org/assets/images/pytorch-logo.png", use_column_width=True)
    col_2.image("https://images4.programmersought.com/878/c8/c8b175f9d26f422afd56a6a20285302e.png", use_column_width=True)
    streamlit.write('''ESRGAN model is implemented in PyTorch.
            This streamlit example uses a FastAPI service as backend.
            Visit this URL at `:8000/docs` for FastAPI documentation.''') # description and instructions

    image = streamlit.file_uploader('insert image')  # image upload widget

    @streamlit.cache
    def process(image, server_url: str):

        m = MultipartEncoder(
            fields={'file': ('filename.jpg', image, 'image/jpeg')}
            )

        r = requests.post(server_url,
                        data=m,
                        headers={'Content-Type': m.content_type},
                        timeout=8000)

        return r


    if streamlit.button('Generate'):

        if image == None:
            streamlit.write("Insert an image!")  # handle case with no image
        else:
            col1, col2 = streamlit.beta_columns(2)
            input_image = process(image, url+endpoint)
            generated_image = Image.open(io.BytesIO(input_image.content)).convert('RGB')
            col1.header("Input Image")
            col1.image(image, use_column_width=True)
            col2.header("Output Image")
            col2.image(generated_image, use_column_width=True)
            # streamlit.image([image, segmented_image], width=300)

if __name__ == '__main__':
    if streamlit._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())