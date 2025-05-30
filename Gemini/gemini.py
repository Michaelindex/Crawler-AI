# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate():
    # Lendo a chave da API do arquivo
    with open('../apis/gemini.key', 'r') as file:
        api_key = file.read().strip()

    client = genai.Client(
        api_key=api_key
    )

    model = "gemini-2.5-flash-preview-04-17"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Você é um assistente que pesquisa dados de médicos. 
Entrada conhecida dos médicos:

  Nome: \"JOSE DE SOUZA GERMINO\"
  CRM: \"5241\"
  UF: \"AL\"
                                
Campos faltantes a buscar de cada médico:
  - Endereço completo (logradouro, número, complemento, bairro, CEP, cidade, estado)
  - Telefone fixo do local de trabalho (até 2 números)
  - Celular (até 2 números)
  - E-mail (até 2 endereços)
**Formato de resposta:** JSON com as chaves  
{\"especialidade_medica\",\"endereco_completo_a1\",\"numero_a1\",\"complemento_a1\",\"bairro_a1\",\"cep_a1\",\"cidade_a1\",\"estado_a1\",\"telefone1\",\"telefone2\",\"celular1\",\"celular2\",\"email1\",\"email2\"}.
## Exemplo de Resposta Esperada (JSON)  
```json
{
  \"especialidade_medica\": \"Cardiologia\",
  \"endereco_completo_a1\": \"Rua das Flores, 123, Sala 45, Centro, São Paulo, SP, 01000-000\",
  \"numero_a1\": \"123\",
  \"complemento_a1\": \"Sala 45\",
  \"bairro_a1\": \"Centro\",
  \"cep_a1\": \"01000000\",
  \"cidade_a1\": \"São Paulo\",
  \"estado_a1\": \"SP\",
  \"telefone1\": \"(11) 1234-5678\",
  \"telefone2\": \"\",
  \"celular1\": \"(11) 98765-4321\",
  \"celular2\": \"\",
  \"email1\": \"fulano.tal@hospital.org\",
  \"email2\": \"\""""),
            ],
        ),
    ]
    tools = [
        types.Tool(google_search=types.GoogleSearch()),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        tools=tools,
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
