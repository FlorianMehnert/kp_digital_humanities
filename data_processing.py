import st_dataset_v1 as ds


def load_and_process_data():
    html = ds.scrape_webpage("https://www.gutenberg.org/files/701/701-h/701-h.htm#chap01")
    content = ds.extract_content(html)
    processed_content = ds.process_text(content)
    return processed_content


def create_gapped_paragraphs(content, mask_rate):
    return [ds.create_gaps(s, mask_rate) for s in content]
