# Import libs
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
import os

# Config API TOKEN
load_dotenv()
api_token = os.environ.get('GROQ_API_KEY')

# Create a LLM
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_retries=2,
    api_key=api_token,
)

"""
Function intended for instruction and core of the agent for travel recommendation based on the requester's interest
"""
def get_travel_recommendation(interest):
    # Add context to model
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Você é um guia turístico que recomenda cidades e países ao usuário.
            Além de contar um pouco sobre a história do local, recomendar restaurantes e lugares turísticos com base no interesse, ou no local escolhido pelo usuário""")
        ]
    )
    
    # First template: suggest country/city based on interest
    template = ChatPromptTemplate.from_template(
        """
        Dado o meu interesse por {interest}, sugira uma cidade ou um país.
        Responda apenas com o nome da cidade ou país sugerido.
        """
    )
    
    # Second template: tell about the country/city
    about_template = ChatPromptTemplate.from_template(
        """
        Conte um pouco sobre a história do país ou cidade {destination}.
        """
    )
    
    # Third template: suggest restaurants
    restaurant_template = ChatPromptTemplate.from_template(
        """
        Sugira os 5 melhores restaurantes do país ou cidade {destination}.
        """
    )
    
    # Fourth template: suggest cultural activities based on interest
    cultural_activities_template = ChatPromptTemplate.from_template(
        """
        Sugira as 5 melhores atividades culturais do país ou cidade {destination} e atividades relacionadas ao interesse do usuário {interest}.
        """
    )
    
    # STEPS
    ## 1. Get destination
    get_destination = template | llm | StrOutputParser()
    
    ## 2. Get about destination
    get_about_detination = about_template | llm | StrOutputParser()
    
    ## 3. Get restaurants
    get_restaurants = restaurant_template | llm | StrOutputParser()
    
    ## 4. Get cultural activities
    get_cultural_activities = cultural_activities_template | llm | StrOutputParser()
    
    ## Chain
    chain = (
        prompt 
        | {"get_destination": get_destination} 
        | {"destination": lambda x: x} # Get just country/city
        | {
            "about": get_about_detination, 
            "restaurants": get_restaurants, 
            "cultural_activities": lambda x: get_cultural_activities.invoke({"interest": interest, "destination": x})
        }
    )
    
    # Invoke chain
    response = chain.invoke({"interest": interest})
    
    # Format output
    formatted_output = (
        f"Sobre o destino:\n{response['about']}\n\n"
        f"Restaurantes recomendados:\n{response['restaurants']}\n\n"
        f"Atividades culturais sugeridas:\n{response['cultural_activities']}"
    )
    
    return formatted_output

# Given that the user knows which destination he wants to go to, then generate an itinerary for the trip
def generate_travel_plan(dest):
    prompt = f"""
    Dado a cidade ou país inserida pelo usuário {dest}, crie um roteiro de viagem detalhado.
    Neste roteiro conte sobre a história da cidade ou país, indique 5 restaurantes imperdiveis, além de selecionar atrações turísticas.
    """
    response = llm.invoke(prompt)
    return response.content

# Process the user input
def process_input(option, user_input):
    if option == "Recomendar um país ou cidade":
        # Check if input format is in string, if not convert
        try:
            return get_travel_recommendation(str(user_input))
        except ValueError as error:
            return f"Erro ao processar a recomendação: {str(error)}"
    elif option == "Já tenho um lugar em mente":
        # Check if input format is in string, if not convert
        try:
            return generate_travel_plan(str(user_input))
        except ValueError as error:
            return f"Erro ao gerar o roteiro: {str(error)}"

# Build frontend
with gr.Blocks() as interface:
    # Dropdown to choise the options
    option = gr.Dropdown(choices=["Recomendar um país ou cidade", "Já tenho um lugar em mente"], label="Escolha uma opção")
    
    # Textbox to user interest
    user_input = gr.Textbox(label="Digite seu interesse ou país/cidade", placeholder="Exemplo: praia, Paris, etc.")
    
    # Submit button
    submit_btn = gr.Button("Gerar Roteiro")
    
    # Output area
    output = gr.Textbox(label="Roteiro de viagem")
    
    # When the user click on submit btn call the process_input function
    submit_btn.click(process_input, inputs=[option, user_input], outputs=output)

if __name__ == __name__:
    interface.launch()