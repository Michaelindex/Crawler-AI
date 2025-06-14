import pandas as pd
import os

def transform_input():
    # Verifica se o arquivo de entrada existe
    if not os.path.exists('input.csv'):
        print("Erro: Arquivo 'input.csv' não encontrado!")
        return
    
    try:
        # Lê o arquivo de entrada
        df = pd.read_csv('input.csv')
        
        # Cria um novo DataFrame com a estrutura esperada
        new_df = pd.DataFrame(columns=[
            'Hash', 'CRM', 'UF', 'Firstname', 'LastName', 'Medical specialty',
            'Endereco Completo A1', 'Address A1', 'Numero A1', 'Complement A1', 'Bairro A1',
            'postal code A1', 'City A1', 'State A1', 'Phone A1', 'Phone A2',
            'Cell phone A1', 'Cell phone A2', 'E-mail A1', 'E-mail A2', 'OPT-IN', 'STATUS', 'LOTE'
        ])
        
        # Função para converter para inteiro apenas se possível
        def safe_int_convert(x):
            try:
                return int(float(x)) if pd.notnull(x) else ''
            except (ValueError, TypeError):
                return str(x) if pd.notnull(x) else ''
        
        # Função para obter valor seguro de uma coluna
        def safe_get_column(df, column_name):
            if column_name in df.columns:
                return df[column_name]
            return pd.Series([''] * len(df))
        
        # Copia apenas os campos que queremos manter
        new_df['Hash'] = ''  # Define Hash como string vazia
        new_df['CRM'] = safe_get_column(df, 'CRM').apply(safe_int_convert)
        new_df['UF'] = safe_get_column(df, 'UF')
        new_df['Firstname'] = safe_get_column(df, 'Firstname')
        new_df['LastName'] = safe_get_column(df, 'LastName')
        new_df['Medical specialty'] = safe_get_column(df, 'Medical specialty')
        
        # Define todos os outros campos como vazios
        new_df['Endereco Completo A1'] = ''
        new_df['Address A1'] = ''
        new_df['Numero A1'] = ''
        new_df['Complement A1'] = ''
        new_df['Bairro A1'] = ''
        new_df['postal code A1'] = ''
        new_df['City A1'] = ''
        new_df['State A1'] = ''
        new_df['Phone A1'] = ''
        new_df['Phone A2'] = ''
        new_df['Cell phone A1'] = ''
        new_df['Cell phone A2'] = ''
        new_df['E-mail A1'] = ''
        new_df['E-mail A2'] = ''
        new_df['OPT-IN'] = ''
        new_df['STATUS'] = ''
        new_df['LOTE'] = ''
        
        # Preenche todos os valores NaN com string vazia
        new_df = new_df.fillna('')
        
        # Salva o arquivo transformado
        new_df.to_csv('input_transformed.csv', index=False)
        print("Arquivo transformado salvo como 'input_transformed.csv'")
        print(f"Total de linhas processadas: {len(new_df)}")
        
    except Exception as e:
        print(f"Erro ao processar o arquivo: {str(e)}")
        print("Colunas encontradas no arquivo de entrada:", df.columns.tolist() if 'df' in locals() else "Nenhuma")

if __name__ == "__main__":
    transform_input() 