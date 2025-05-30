# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types
import json
import re
import time  # Adicionado para o delay

def extract_json_from_text(text):
    """Extrai o JSON da resposta do modelo."""
    try:
        # Procura por um bloco JSON na resposta
        json_match = re.search(r'```json\s*({[\s\S]*?})\s*```', text)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        return {}
    except Exception as e:
        print(f"Erro ao extrair JSON: {e}")
        return {}

def build_prompt(current_data):
    """Constrói o prompt com os dados acumulados."""
    known_data = "Entrada conhecida dos médicos:\n"
    for key, value in current_data.items():
        if value:  # Só inclui campos não vazios
            known_data += f"  {key}: \"{value}\"\n"
    
    return f"""Você é um assistente que pesquisa dados de médicos. 
{known_data}
                                
Campos faltantes a buscar de cada médico:
  - Endereço completo (logradouro, número, complemento, bairro, CEP, cidade, estado)
  - Telefone fixo do local de trabalho (até 2 números)
  - Celular (até 2 números)
  - E-mail (até 2 endereços)
**Formato de resposta:** JSON com as chaves  
{{"especialidade_medica","endereco_completo_a1","numero_a1","complemento_a1","bairro_a1","cep_a1","cidade_a1","estado_a1","telefone1","telefone2","celular1","celular2","email1","email2"}}.
## Exemplo de Resposta Esperada (JSON)  
```json
{{
  "especialidade_medica": "Cardiologia",
  "endereco_completo_a1": "Rua das Flores, 123, Sala 45, Centro, São Paulo, SP, 01000-000",
  "numero_a1": "123",
  "complemento_a1": "Sala 45",
  "bairro_a1": "Centro",
  "cep_a1": "01000000",
  "cidade_a1": "São Paulo",
  "estado_a1": "SP",
  "telefone1": "(11) 1234-5678",
  "telefone2": "",
  "celular1": "(11) 98765-4321",
  "celular2": "",
  "email1": "fulano.tal@hospital.org",
  "email2": ""
}}```"""

def generate():
    # Lendo a chave da API do arquivo
    with open('../apis/gemini.key', 'r') as file:
        api_key = file.read().strip()

    client = genai.Client(
        api_key=api_key
    )

    model = "gemini-2.5-flash-preview-04-17"
    
    # Dados iniciais
    current_data = {
        "Nome": "JOSE DE SOUZA GERMINO",
        "CRM": "5241",
        "UF": "AL",
        "especialidade_medica": "",
        "endereco_completo_a1": "",
        "numero_a1": "",
        "complemento_a1": "",
        "bairro_a1": "",
        "cep_a1": "",
        "cidade_a1": "",
        "estado_a1": "",
        "telefone1": "",
        "telefone2": "",
        "celular1": "",
        "celular2": "",
        "email1": "",
        "email2": ""
    }

    # Loop de 6 iterações
    for iteration in range(6):
        print(f"\n=== Iteração {iteration + 1} ===")
        
        # Delay incremental de 7 segundos, com 45 segundos na última iteração
        if iteration > 0:
            if iteration == 5:  # Última iteração
                delay = 45
            else:
                delay = 7 * iteration
            print(f"Aguardando {delay} segundos antes da próxima requisição...")
            time.sleep(delay)
        
        # Construir o prompt com os dados atuais
        prompt_text = build_prompt(current_data)
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt_text),
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

        # Coletar a resposta completa
        full_response = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            print(chunk.text, end="")
            full_response += chunk.text

        # Extrair e atualizar dados
        new_data = extract_json_from_text(full_response)
        for key, value in new_data.items():
            if value and (not current_data.get(key) or len(value) > len(current_data.get(key, ""))):
                current_data[key] = value

    # Imprimir o resultado final
    print("\n\n=== Resultado Final ===")
    print(json.dumps(current_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    generate()
