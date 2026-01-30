import pandas as pd

# --- Функция расширения POS ---
def expand_pos_column(df, column_name='POS_Dependencies'):
    expanded_rows = []

    for _, row in df.iterrows():
        pos_str = row[column_name]
        pos_dict = {}

        if pd.notna(pos_str) and pos_str != '':
            items = pos_str.split(', ')
            for item in items:
                if ': ' in item:
                    pos, count = item.split(': ')
                    pos_dict[pos] = int(count)

        # Добавляем оригинальную строку, чтобы не потерять данные
        combined = row.to_dict()
        combined.update(pos_dict)
        expanded_rows.append(combined)

    return pd.DataFrame(expanded_rows)