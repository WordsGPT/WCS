import re

with open('mobydick.html', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('<title>WCS Moby-Dick Comparison</title>', '<title>WCS Language Comparison</title>')
content = content.replace('<h1>WCS Comparison</h1>', '<h1>WCS Language Comparison</h1>')

content = content.replace('''<button type="button" data-dataset="mobydick.csv" class="active">Moby-Dick</button>
          <button type="button" data-dataset="western_lands.csv">Western Lands</button>
          <button type="button" data-dataset="FortunayJavinta.csv">Fortunata</button>
          <button type="button" data-dataset="all">All</button>''', 
'''<button type="button" data-dataset="spanish.csv" class="active">Spanish</button>
          <button type="button" data-dataset="english.csv">English</button>
          <button type="button" data-dataset="all">Both</button>''')

content = content.replace('dataset: "mobydick.csv"', 'dataset: "spanish.csv"')

content = content.replace('["mobydick.csv", "western_lands.csv", "FortunayJavinta.csv"]', '["spanish.csv", "english.csv"]')

content = content.replace('Comparing Moby-Dick, Western Lands, and Fortunata y Jacinta', 'Comparing Spanish (solid) vs English (dotted)')
content = content.replace('Comparing Moby-Dick (solid/dash) vs Western Lands (dotted) vs Fortunata (long-dash)', 'Comparing Spanish (solid) vs English (dotted)')

content = content.replace('Dataset: ${row.dataset === "mobydick.csv" ? "Moby-Dick" : (row.dataset === "western_lands.csv" ? "Western Lands" : "Fortunata y Jacinta")}', 'Dataset: ${row.dataset === "spanish.csv" ? "Spanish" : "English"}')

draw_series_old = '''        const isWL = dataset === "western_lands.csv";
        const isFJ = dataset === "FortunayJavinta.csv";
        const path = values.map((row, index) => `${index ? "L" : "M"}${x(xValue(row)).toFixed(2)},${y(row[state.metric]).toFixed(2)}`).join(" ");
        const pathEl = document.createElementNS("http://www.w3.org/2000/svg", "path");
        pathEl.setAttribute("d", path);
        pathEl.setAttribute("class", `series ${variantClass(model)}`);
        pathEl.setAttribute("stroke", colorFor(model));
        if (isWL) {
           pathEl.setAttribute("stroke-dasharray", variantClass(model) === "base" ? "2 3" : "8 3 2 3");
           pathEl.setAttribute("stroke-width", "1.8");
        } else if (isFJ) {
           pathEl.setAttribute("stroke-dasharray", variantClass(model) === "base" ? "5 5" : "10 5 5 5");
           pathEl.setAttribute("stroke-width", "1.8");
        }
        svg.append(pathEl);
        values.forEach(row => {
          const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          circle.setAttribute("cx", x(xValue(row)));
          circle.setAttribute("cy", y(row[state.metric]));
          circle.setAttribute("r", (isWL || isFJ) ? (variantClass(model) === "base" ? 3 : 3.8) : (variantClass(model) === "base" ? 4 : 4.8));'''

draw_series_new = '''        const isEnglish = dataset === "english.csv";
        const path = values.map((row, index) => `${index ? "L" : "M"}${x(xValue(row)).toFixed(2)},${y(row[state.metric]).toFixed(2)}`).join(" ");
        const pathEl = document.createElementNS("http://www.w3.org/2000/svg", "path");
        pathEl.setAttribute("d", path);
        pathEl.setAttribute("class", `series ${variantClass(model)}`);
        pathEl.setAttribute("stroke", colorFor(model));
        if (isEnglish) {
           pathEl.setAttribute("stroke-dasharray", variantClass(model) === "base" ? "2 3" : "8 3 2 3");
           pathEl.setAttribute("stroke-width", "1.8");
        }
        svg.append(pathEl);
        values.forEach(row => {
          const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          circle.setAttribute("cx", x(xValue(row)));
          circle.setAttribute("cy", y(row[state.metric]));
          circle.setAttribute("r", (isEnglish) ? (variantClass(model) === "base" ? 3 : 3.8) : (variantClass(model) === "base" ? 4 : 4.8));'''

content = content.replace(draw_series_old, draw_series_new)

content = content.replace('<code>mobydick.csv</code>', '<code>spanish.csv</code>')

with open('language.html', 'w', encoding='utf-8') as f:
    f.write(content)
