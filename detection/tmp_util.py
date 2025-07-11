from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def chunk_rows_by_length(rows_data, max_chars=2000):
    if not rows_data:
        return

    batch = []
    current_len = 0
    keys = rows_data[0].keys()

    for row in rows_data:
        # Convert row to a JSON-like string representation
        row_str = '{' + ', '.join(f'"{key}": "{row[key]}"' for key in keys) + '}'
        # print(row_str, type(row_str))

        if current_len + len(row_str) + 2 > max_chars and batch:
            yield batch
            batch = []
            current_len = 0
        batch.append(row_str)
        current_len += len(row_str) + 2
    if batch:
        yield batch





