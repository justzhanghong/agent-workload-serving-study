#!/usr/bin/env python3
"""Build and inspect the PDF directly from the released report Markdown."""
import base64
import hashlib
import html
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import quote, urlsplit

import fitz
from markdown_it import MarkdownIt
from pypdf import PdfReader, PdfWriter
from report_browser import ROOT, ReportBrowser

SOURCE = ROOT / "paper/report.zh-CN.md"
TARGET = SOURCE.with_suffix(".pdf")
PUBLIC = "https://github.com/justzhanghong/agent-workload-serving-study/blob/main/"

CSS = """
@page { size: A4; margin: 17mm 16mm 19mm; }
* { box-sizing: border-box; }
html { font-family: 'WenQuanYi Micro Hei','Noto Sans CJK SC',sans-serif;
       font-size: 10.1pt; color: #263040; }
body { margin: 0; line-height: 1.64; }
h1 { font-size: 23pt; line-height: 1.5; color: #AF2E3D; margin: 1em 0; }
h2 { font-size: 16pt; color: #AF2E3D; margin: 1.4em 0 .6em; }
h3 { font-size: 12pt; color: #8e3446; margin: 1.2em 0 .5em; }
h1,h2,h3,h4 { break-after: avoid; }
p { margin: .65em 0; orphans: 3; widows: 3; overflow-wrap: anywhere; }
a { color: #97384f; text-decoration: none; overflow-wrap: anywhere; }
li { margin: .4em 0; }
ul,ol { padding-left: 1.8em; }
table { border-collapse: collapse; width: 100%; font-size: 8.1pt;
        line-height: 1.5; margin: 1em 0; table-layout: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th,td { text-align: left; padding: 5px 5px; border: .5px solid #d6c7cd;
        overflow-wrap: anywhere; vertical-align: top; }
th { background: #f3e9ed; color: #722738; }
tbody tr:nth-child(even) { background: #fcf9fa; }
code,pre { font-family: 'DejaVu Sans Mono','WenQuanYi Micro Hei',monospace;
           font-size: 8.8pt; }
pre { background: #f7f4f6; border-left: 2px solid #CB5979; padding: 10px;
      white-space: pre-wrap; overflow-wrap: anywhere; break-inside: avoid; }
figure { margin: 1em 0; break-inside: avoid; }
figure img { display: block; max-width: 100%; max-height: 203mm; width: 100%;
             object-fit: contain; }
figcaption { font-size: 8.5pt; color: #5d626c; line-height: 1.5; margin: .5em 0; }
.toc { break-after: page; margin-top: 2em; }
.toc h2 { margin: 0 0 1em; }
.toc ol { padding-left: 0; list-style: none; }
.toc li { border-bottom: .5px solid #eadce1; padding: .35em 0; }
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build_html(text):
    md = MarkdownIt("commonmark", {"html":False}).enable("table")
    tokens = md.parse(text)
    headings, figures = [], {}
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            title = tokens[i+1].content
            ident = "section-"+str(len(headings)+1)
            token.attrSet("id", ident)
            headings.append((token.tag, ident, title))
        for child in token.children or []:
            if child.type == "image":
                source = (SOURCE.parent / child.attrGet("src")).resolve()
                source.relative_to(ROOT)
                vector = source.with_suffix(".svg")
                if vector.exists(): source = vector
                data = source.read_bytes()
                figures[str(source.relative_to(ROOT))] = sha(data)
                mime = "image/svg+xml" if source.suffix == ".svg" else "image/png"
                child.attrSet("src", "data:"+mime+";base64,"+base64.b64encode(data).decode())
            elif child.type == "link_open":
                target = child.attrGet("href")
                if not urlsplit(target).scheme and not target.startswith("#"):
                    path,sep,fragment = target.partition("#")
                    relative = (SOURCE.parent/path).resolve().relative_to(ROOT).as_posix()
                    child.attrSet("href", PUBLIC+quote(relative)+(sep+fragment if sep else ""))
    body = md.renderer.render(tokens, md.options, {})
    body = re.sub(r'<p>(<img [^>]+>)</p>\s*<p>(图[\s\S]*?)</p>',
                  r'<figure>\1<figcaption>\2</figcaption></figure>', body)
    # Only top-level sections are listed in print; all headings become bookmarks.
    toc = '<nav class="toc"><h2>目录</h2><ol>'+"".join(
        f'<li><a href="#{ident}">{html.escape(title)}</a></li>'
        for tag,ident,title in headings if tag == "h2")+'</ol></nav>'
    position = body.index('<h2 ')
    body = body[:position]+toc+body[position:]
    title = headings[0][2]
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body>'
            +body+'</body></html>'), figures, title, headings


def main():
    source_bytes = SOURCE.read_bytes()
    document, figures, title, headings = build_html(source_bytes.decode("utf-8"))
    delivery = ROOT / ".delivery"
    delivery.mkdir(exist_ok=True)
    rendered = delivery / "report.html"
    rendered.write_text(document, encoding="utf-8")
    with ReportBrowser() as browser:
        raw = browser.pdf(rendered)
        version = browser.version
    writer = PdfWriter(clone_from=BytesIO(raw))
    writer.add_metadata({"/Title":title,"/Author":"zhanghong",
                         "/Subject":"Coding / Work Agent serving evidence, decode-chip literature and planned bottleneck experiments",
                         "/SourceMarkdownSHA256":sha(source_bytes)})
    with TARGET.open("wb") as output: writer.write(output)
    reader = PdfReader(TARGET)
    extracted = fitz.open(TARGET)
    page_text = [p.get_text() for p in extracted]
    full_text = "\n".join(page_text)
    compact = re.sub(r"\s+", "", full_text)
    required = ["摘要","1.引言","2.相关工作","Coding/Work",
                "TraceLab","Copilot","CacheWise","Continuum","关键路径","状态生命周期",
                "3.方法","4.实验","5.实验结果","结果解读：","6.讨论与局限","参考文献",
                "PPD","AMPD","HeteroPanacea","LLMCompass","3.6.面向decode芯片",
                "E1：Decode算子","E2：KV容量","E3：增量prefill","E4：经校准","待执行"]
    missing = [word for word in required if word not in compact]
    if missing: raise AssertionError("Missing PDF content: "+str(missing))
    blank = [i+1 for i,text in enumerate(page_text) if len(re.sub(r"\s+", "", text)) < 20]
    if blank: raise AssertionError("Unexpected blank PDF pages: "+str(blank))
    out_of_bounds = []
    for index,page in enumerate(extracted):
        for word in page.get_text("words"):
            x0,y0,x1,y1 = word[:4]
            if x0 < -1 or y0 < -1 or x1 > page.rect.width+1 or y1 > page.rect.height+1:
                out_of_bounds.append(index+1)
    if out_of_bounds: raise AssertionError("Text outside PDF pages: "+str(sorted(set(out_of_bounds))))
    uris = [str(a.get_object().get("/A",{}).get("/URI", ""))
            for page in reader.pages for a in page.get("/Annots",[])]
    published_text = "\n".join(uris)+str(reader.metadata)+full_text
    if re.search(r"file://|/nfs/|/home/|/data/ssd|C:\\Users",published_text):
        raise AssertionError("Private/local paths in PDF")
    def count_outline(items):
        return sum(count_outline(x) if isinstance(x,list) else 1 for x in items)
    outline_count = count_outline(reader.outline)
    if outline_count < len(headings):
        raise AssertionError("Missing PDF heading bookmarks")
    preview = delivery / "pdf-preview"
    preview.mkdir(exist_ok=True)
    pages = sorted(set([0,1,2,len(extracted)-1]+[
        i for i,text in enumerate(page_text)
        if any(key in re.sub(r"\s+","",text) for key in
               ("图R1：","图R2：","图R3：","图1：","图E1：","结果解读：",
                "2.10.面向decode芯片","3.6.面向decode芯片","3.6.4.E3","3.6.5.E4"))]))
    for number in pages:
        extracted[number].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(preview/f"page-{number+1:02d}.png")
    manifest = {"date":"2026-10-20","source":"paper/report.zh-CN.md",
        "sourceSha256":sha(source_bytes),"pdf":"paper/report.zh-CN.pdf",
        "pdfSha256":sha(TARGET.read_bytes()),"embeddedFiguresSha256":figures,
        "pageCount":len(reader.pages),"outlineCount":outline_count,
        "browser":version,"renderer":"MarkdownIt + Chrome printToPDF + pypdf metadata",
        "checks":{"requiredTextPresent":True,"noBlankPages":True,
                  "textWithinPages":True,"noPrivatePathsInTextLinksMetadata":True},
        "note":"Source Markdown and SVG hashes establish this build correspondence; not a cross-platform byte-reproducibility claim."}
    (ROOT/"paper/report_pdf.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"pages":manifest["pageCount"],"bookmarks":outline_count,
        "figures":len(figures),"pdfBytes":TARGET.stat().st_size,
        "previewPages":[i+1 for i in pages],"checks":manifest["checks"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
