from __future__ import annotations

from pathlib import Path

import fitz


def build_sample_plan(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    page = doc.new_page(width=1191, height=842)

    page.draw_rect(fitz.Rect(30, 30, 1161, 812), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(160, 160), fitz.Point(700, 160), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(160, 260), fitz.Point(700, 260), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(160, 360), fitz.Point(700, 360), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(250, 120), fitz.Point(250, 420), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(450, 120), fitz.Point(450, 420), color=(0, 0, 0), width=1)

    page.insert_text((175, 145), "LIGNE PRINCIPALE DN200", fontsize=15)
    page.insert_text((470, 245), "POMPE P-101", fontsize=13)
    page.insert_text((320, 345), "VANNE V-204", fontsize=13)

    page.insert_text((730, 120), "1 - DEPART VERS ATELIER", fontsize=11)
    page.insert_text((730, 150), "2 - RETOUR VERS LOCAL TECHNIQUE", fontsize=11)
    page.insert_text((730, 180), "3 - CAPTEUR PT-011", fontsize=11)

    page.insert_text((80, 520), "NOTES:", fontsize=12)
    page.insert_text((80, 545), "NOTE 1: TOUS LES REPERES SONT EN MM", fontsize=10)
    page.insert_text((80, 565), "NOTE 2: RESPECTER LA REVISION B", fontsize=10)

    page.draw_rect(fitz.Rect(650, 500, 1110, 720), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(650, 540), fitz.Point(1110, 540), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(730, 500), fitz.Point(730, 720), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(820, 500), fitz.Point(820, 720), color=(0, 0, 0), width=1)

    page.insert_text((665, 525), "ITEM", fontsize=10)
    page.insert_text((748, 525), "QTY", fontsize=10)
    page.insert_text((835, 525), "DESCRIPTION", fontsize=10)
    page.insert_text((665, 565), "1", fontsize=10)
    page.insert_text((748, 565), "2", fontsize=10)
    page.insert_text((835, 565), "PIPE DN200", fontsize=10)
    page.insert_text((665, 595), "2", fontsize=10)
    page.insert_text((748, 595), "1", fontsize=10)
    page.insert_text((835, 595), "POMPE P-101", fontsize=10)
    page.insert_text((665, 625), "3", fontsize=10)
    page.insert_text((748, 625), "1", fontsize=10)
    page.insert_text((835, 625), "VANNE V-204", fontsize=10)

    page.draw_rect(fitz.Rect(780, 720, 1130, 805), color=(0, 0, 0), width=1)
    page.insert_text((800, 742), "DRAWING TITLE", fontsize=10)
    page.insert_text((800, 758), "PLAN RESEAU HYDRAULIQUE", fontsize=10)
    page.insert_text((800, 774), "SHEET A1  REV B  SCALE 1:50", fontsize=10)
    page.insert_text((800, 790), "DWG H-001  CLIENT DEMO", fontsize=10)

    doc.save(output_path)
    doc.close()


if __name__ == "__main__":
    build_sample_plan(Path("samples/demo_plan.pdf"))
