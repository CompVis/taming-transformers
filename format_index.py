import math

rowdivs = ["""<div class="6u 12u$(xsmall)">\n""",
           """<div class="6u$ 12u$(xsmall)">\n"""]

def make_row(*divs):
    row = """<div class="row 150%">\n"""
    for i, div in enumerate(divs):
        row += rowdivs[i] + div + "</div>\n"
    row += """</div>\n"""
    return row

if __name__ == "__main__":
    with open("listofdivs.template", "r") as f:
        list_of_divs = f.read().split("==========")

    n_cells = len(list_of_divs)
    n_cols = 2
    n_rows = math.ceil(n_cells / n_cols)
    content = ""
    for i in range(n_rows):
        cells = list_of_divs[i*n_cols:(i+1)*n_cols]
        content += make_row(*cells)

    with open("index.template", "r") as f:
        index_template = f.read()
    index = index_template.replace("__TEMPLATE_STRING__", content)
    with open("index.html", "w") as f:
        f.write(index)
