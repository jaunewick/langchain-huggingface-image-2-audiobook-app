from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv(find_dotenv())

# img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text

# llm
def generate_story(scenario):
    template = """
    You are a great audiobook narrator and story teller;
    You can generate a short story based on an interesting narrative, the story should be no more than 50 words;

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    #langchain-huggingface
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=100,
        do_sample=False,
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
    )
    story = llm_chain.predict(scenario=scenario)
    print(story)
    return story

scenario = img2text("photo.png")
story = generate_story(scenario)

#text2speech

