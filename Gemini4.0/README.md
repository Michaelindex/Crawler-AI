# Gemini4.0

## Estrutura do Arquivo de Entrada (input.csv)

O arquivo de entrada deve conter as seguintes colunas:

| Coluna | Descrição | Obrigatório |
|--------|-----------|-------------|
| CRM | Número do CRM do médico | Sim |
| UF | Estado do CRM | Sim |
| Firstname | Nome do médico | Sim |
| LastName | Sobrenome do médico | Sim |
| Medical specialty | Especialidade médica | Sim |

### Observações:
- Todos os outros campos serão preenchidos automaticamente pelo sistema
- Os campos obrigatórios não podem estar vazios
- O CRM deve ser um número válido
- A UF deve ser uma sigla válida de estado brasileiro

### Exemplo de input.csv:
```csv
CRM,UF,Firstname,LastName,Medical specialty
12345,SP,João,Silva,Pediatria
67890,RJ,Maria,Santos,Clínico Geral
```

## Como Usar

1. Prepare seu arquivo `input.csv` com os campos obrigatórios
2. Coloque o arquivo na pasta do Gemini4.0
3. Execute o script principal
4. O sistema irá processar os dados e gerar o arquivo de saída 