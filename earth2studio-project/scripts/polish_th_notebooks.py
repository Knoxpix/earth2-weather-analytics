#!/usr/bin/env python3
"""Polish Thai notebook translations with repo-specific replacements."""

from __future__ import annotations

import json
from pathlib import Path


GLOBAL_REPLACEMENTS = [
    ("# การรันการอนุมานเชิงกำหนด", "# การรัน Deterministic Inference"),
    ("# กำลังเรียกใช้การอนุมานการวินิจฉัย", "# การรัน Diagnostic Inference"),
    ("# จุดตรวจตระการตาขนาดใหญ่ (HENS)", "# Huge Ensembles (HENS)"),
    ("# ขยาย Prognostic Model", "# การขยาย Prognostic Model"),
    ("การพยากรณ์โรค", "การพยากรณ์"),
    ("การโพสต์ประมวลผล", "การทำ post-process"),
    ("ขั้นตอนสุดท้ายคือการทำ post-processผลลัพธ์ของเรา", "ขั้นตอนสุดท้ายคือการนำผลลัพธ์มาทำ post-process ต่อ"),
    ("ขั้นตอนสุดท้ายคือการทำ post-processผลลัพธ์ของเรา Cartopy เป็นไลบรารีที่ยอดเยี่ยมสำหรับการพล็อต", "ขั้นตอนสุดท้ายคือการนำผลลัพธ์มาทำ post-process ต่อ Cartopy เป็นไลบรารีที่เหมาะมากสำหรับการพล็อต"),
    ("แพ็คเกจโมเดล", "model package"),
    ("โหลดแพ็คเกจโมเดลเริ่มต้นซึ่งดาวน์โหลดจุดตรวจสอบจาก NGC", "โหลด model package เริ่มต้น ซึ่งจะดาวน์โหลด checkpoint จาก NGC"),
    ("โหลดแพ็คเกจโมเดลเริ่มต้นซึ่งดาวน์โหลด checkpoint จาก GCP", "โหลด model package เริ่มต้น ซึ่งจะดาวน์โหลด checkpoint จาก GCP"),
    ("โหลดแพ็คเกจโมเดลเริ่มต้น (ดาวน์โหลด checkpoint จาก HuggingFace)", "โหลด model package เริ่มต้น (ดาวน์โหลด checkpoint จาก HuggingFace)"),
    ("จุดตรวจสอบ", "checkpoint"),
    ("จุดตรวจ", "checkpoint"),
    ("checkpoint ของโมเดล", "checkpoint ของโมเดล"),
    ("การวางแผน", "การพล็อต"),
    ("โปรดสังเกตว่าฟังก์ชัน", "สังเกตว่าฟังก์ชัน"),
    ("เพื่อโต้ตอบกับข้อมูลที่เก็บไว้", "สำหรับใช้เข้าถึงและจัดการข้อมูลที่เก็บไว้"),
    ("workflow ทั้งหมดภายใน Earth2Studio จำเป็นต้องมีส่วนประกอบที่สร้างขึ้น\nมอบให้พวกเขา", "workflow ทุกตัวภายใน Earth2Studio จำเป็นต้องได้รับคอมโพเนนต์ที่สร้างไว้ล่วงหน้า\nแล้วส่งเข้าไปให้ใช้งาน"),
    ("ไร้สัญชาติ", "แบบ stateless"),
    ("วิธีการเพาะเมล็ด", "วิธีการ seed"),
    ("โมเดลฐานการพยากรณ์", "โมเดลพยากรณ์พื้นฐาน"),
    ("ทั่วไปเท่านั้น", "แบบ conventional เท่านั้น"),
    ("ไฟล์เก็บถาวรการเล่นซ้ำ", "ไฟล์เก็บถาวร reanalysis/replay"),
    ("ที่เก็บถาวรการเล่นซ้ำ", "ที่เก็บถาวร reanalysis/replay"),
    ("โมดูลคบเพลิง", "โมดูล PyTorch"),
    ("ผลบวกจะเป็นไปตามปกติ\nเสียงไปยังสนามลมพื้นผิวทุกขั้นตอน", "การเพิ่ม noise แบบ Gaussian\nเข้าไปในสนามลมผิวพื้นในทุก step"),
    ("การอนุมาน Earth-2\nStudio", "Earth-2 Inference\nStudio"),
    ("forecast ที่กำหนดพื้นฐาน", "deterministic forecast เบื้องต้น"),
    ("ขั้นตอนการทำงานอนุมานเชิงกำหนดพื้นฐาน", "workflow พื้นฐานสำหรับ deterministic inference"),
    ("พื้นฐาน deterministic inference workflow", "workflow พื้นฐานสำหรับ deterministic inference"),
    ("การพยากรณ์ขั้นพื้นฐาน + การวินิจฉัย inference workflow", "workflow พื้นฐานสำหรับการใช้โมเดลพยากรณ์ร่วมกับโมเดลวินิจฉัย"),
    ("ฟิลด์บน projections ของทรงกลม", "ฟิลด์ข้อมูลบน projection ของทรงกลม"),
    ("ฟิลด์บน projection ของทรงกลม", "ฟิลด์ข้อมูลบน projection ของทรงกลม"),
    ("ตาราง\n", "กริด\n"),
    ("แบบจำลองการดูดซึมข้อมูล HealDA", "โมเดลการดูดซึมข้อมูล HealDA"),
    ("ตัวอย่างนี้สาธิตวิธีใช้แบบจำลองการดูดซึมข้อมูล HealDA เพื่อสร้าง", "ตัวอย่างนี้สาธิตวิธีใช้โมเดลการดูดซึมข้อมูล HealDA เพื่อสร้าง"),
    ("แบบจำลองการวินิจฉัย", "โมเดลการวินิจฉัย"),
    ("แบบจำลอง", "โมเดล"),
    ("พล็อตและเปรียบเทียบค่าเฉลี่ย ensemble", "พล็อตและเปรียบเทียบค่าเฉลี่ยของ ensemble"),
    ("เข้าสู่ forecast", "ใน forecast"),
    ("20 ขั้นตอน forecast", "20 forecast steps"),
    ("สำหรับ forecast", "สำหรับการพยากรณ์"),
    ("เริ่มต้นด้วยการพล็อตการสะท้อนแสง", "เริ่มจากการพล็อตค่าการสะท้อนเรดาร์"),
    ("ปริมาณฝนทั้งหมดที่คาดการณ์ไว้", "ปริมาณฝนรวมที่คาดการณ์ได้"),
    ("แบบกำหนดเอง\nworkflow", "workflow แบบกำหนดเอง"),
    ("สร้างการพยากรณ์", "สร้างโมเดลพยากรณ์"),
    ("การใช้ Prognostic Model แบบกำหนดเอง", "การใช้งาน Prognostic Model แบบกำหนดเอง"),
    ("การใช้ Diagnostic Model แบบกำหนดเอง", "การใช้งาน Diagnostic Model แบบกำหนดเอง"),
    ("DataFrames แบบธรรมดา", "DataFrame ของข้อมูลสังเกตแบบ conventional"),
    ("การผสมผสานการสังเกตที่แตกต่างกัน", "การผสมชุดข้อมูลสังเกตที่แตกต่างกัน"),
    ("ฟิลด์ส่วนกลางที่หลอมรวม", "ฟิลด์วิเคราะห์ที่ได้จากการหลอมรวม"),
    ("การวิเคราะห์สภาพอากาศทั่วโลกบนตาราง HEALPix จากแหล่งกำเนิดแบบกระจัดกระจาย (แบบทั่วไป) และ\nการสังเกตการแผ่รังสีของดาวเทียมที่มาจากไฟล์เก็บถาวรการเล่นซ้ำของ NOAA UFS", "การวิเคราะห์สภาพอากาศทั่วโลกบนกริด HEALPix จากข้อมูลสังเกตภาคพื้นแบบกระจัดกระจาย (conventional observations) และการสังเกตการแผ่รังสีจากดาวเทียมที่มาจากไฟล์เก็บถาวรของ NOAA UFS"),
    ("มีการเปรียบเทียบการทดสอบ 3 ครั้ง: การสังเกตการณ์ทั่วไปเท่านั้น การสังเกตด้วยดาวเทียมเท่านั้น", "มีการเปรียบเทียบ 3 กรณีทดสอบ: ใช้ข้อมูลสังเกตแบบ conventional เท่านั้น ใช้ข้อมูลดาวเทียมเท่านั้น"),
    ("HealDA คือโมเดลการดูดซึมข้อมูลบนเครือข่ายประสาทแบบ statelessที่นำเข้า", "HealDA คือโมเดลการดูดซึมข้อมูลแบบโครงข่ายประสาทที่ทำงานแบบ stateless โดยรับอินพุตเป็น"),
    ("แบบธรรมดา (วิทยุ, สถานีพื้นผิว, GPS-RO ฯลฯ) และการแผ่รังสีของดาวเทียม\nการสังเกตและสร้างการวิเคราะห์สภาพอากาศทั่วโลกเพียงครั้งเดียวในระดับ HEALPix-6\nกริด", "ข้อมูลสังเกตแบบ conventional (เช่น radiosonde, สถานีผิวพื้น, GPS-RO) และการสังเกตการแผ่รังสีจากดาวเทียม แล้วสร้างฟิลด์วิเคราะห์สภาพอากาศทั่วโลกแบบ one-shot บนกริด HEALPix-6"),
    ("ข้อกำหนด API ของแบบจำลองการพยากรณ์", "ข้อกำหนดของ API สำหรับ Prognostic Model"),
    ("ตัวอย่างนี้จะสาธิตวิธีการขยาย Earth2Studio โดยการนำแบบกำหนดเองไปใช้\nPrognostic Model และรันใน workflow ทั่วไป", "ตัวอย่างนี้จะแสดงวิธีขยาย Earth2Studio โดยการสร้าง Prognostic Model แบบกำหนดเอง แล้วนำไปรันใน workflow ทั่วไปของระบบ"),
    ("## การพยากรณ์แบบกำหนดเอง", "## Prognostic Model แบบกำหนดเอง"),
    ("สามารถใช้เพื่อช่วยได้\nแนะนำ API ที่จำเป็นซึ่งจำเป็นในการสร้างการพยากรณ์แบบกำหนดเองของเราเองได้สำเร็จ", "ซึ่งช่วยกำหนด API ที่จำเป็นสำหรับการสร้างโมเดลพยากรณ์แบบกำหนดเองของเราเอง"),
    ("ในตัวอย่างนี้ เรามาสร้างการพยากรณ์แบบง่ายๆ ที่คาดการณ์ว่าผลบวกจะเป็นไปตามปกติ\nเสียงไปยังสนามลมพื้นผิวทุกขั้นตอน แม้ว่าจะไม่สามารถใช้งานได้จริง แต่ก็ควรทำเช่นนี้\nสาธิต API ที่เราจำเป็นต้องนำไปใช้สำหรับการพยากรณ์", "ในตัวอย่างนี้ เราจะสร้างโมเดลอย่างง่ายที่เพิ่ม Gaussian noise เข้าไปในสนามลมผิวพื้นในทุก step แม้จะไม่ใช่โมเดลที่ใช้งานจริง แต่เหมาะสำหรับสาธิต API ที่จำเป็นสำหรับ Prognostic Model"),
    ("เริ่มต้นด้วยตัวสร้าง โมเดลการพยากรณ์ควรเป็นโมดูล PyTorch\nโมเดลจำเป็นต้องมีวิธี :py:obj:`to(device)` ที่สามารถย้ายโมเดลระหว่างกันได้\nอุปกรณ์ที่แตกต่างกัน หากโมเดลของคุณคือ PyTorch สิ่งนี้จะเป็นเรื่องง่าย", "เริ่มจาก constructor ก่อน โดยทั่วไปโมเดลพยากรณ์ควรเป็นโมดูล PyTorch และควรมีเมธอด :py:obj:`to(device)` เพื่อย้ายโมเดลข้ามอุปกรณ์ต่าง ๆ ได้ หากโมเดลของคุณเขียนด้วย PyTorch อยู่แล้ว ส่วนนี้จะทำได้ไม่ยาก"),
    ("checkpoint Huge Ensembles (HENS) หลายตัวพื้นฐาน inference workflow", "workflow พื้นฐานสำหรับการใช้งาน Huge Ensemble checkpoints หลายชุด"),
    ("ตัวอย่างนี้เป็นตัวอย่างพื้นฐานในการโหลด Huge Ensemble checkpoints เพื่อดำเนินการ\nensemble inference.\nโน้ตบุ๊คนี้มีจุดมุ่งหมายเพื่อแสดงพื้นฐานของการใช้งาน multi-checkpoint workflow\nจากส่วนประกอบ Earth2Studio", "ตัวอย่างนี้แสดงวิธีโหลด Huge Ensemble checkpoints เพื่อรัน ensemble inference โดยมีจุดมุ่งหมายเพื่อสาธิตพื้นฐานของ multi-checkpoint workflow จากคอมโพเนนต์ของ Earth2Studio"),
    ("เราขอแนะนำให้ผู้ใช้ทำความคุ้นเคยกับข้อจำกัดใบอนุญาตของสิ่งนี้\ncheckpoint ของโมเดล", "ขอแนะนำให้ผู้ใช้ตรวจสอบข้อจำกัดด้านใบอนุญาตของ checkpoint โมเดลนี้ก่อนใช้งาน"),
    ("สำหรับ HENS workflow ที่สมบูรณ์ เราขอแนะนำให้ผู้ใช้ดูสูตร HENS\nซึ่งนำเสนอโซลูชั่นแบบ end-to-end เพื่อใช้ประโยชน์จาก HENS สำหรับการวิเคราะห์ขั้นปลาย เช่น\nการติดตามพายุหมุนเขตร้อน:", "สำหรับ HENS workflow แบบครบถ้วน เราแนะนำให้ดู recipe ของ HENS ซึ่งเป็นโซลูชันแบบ end-to-end สำหรับนำ HENS ไปใช้กับงานวิเคราะห์ขั้นปลาย เช่น การติดตามพายุหมุนเขตร้อน:"),
    ("- วิธีโหลด HENS checkpoints ด้วยแพ็คเกจโมเดลแบบกำหนดเอง", "- วิธีโหลด HENS checkpoints ด้วย model package แบบกำหนดเอง"),
    ("- วิธีโหลดวิธีก่อกวน HENS", "- วิธีโหลดวิธี perturbation ของ HENS"),
    ("- วิธีการเห็นภาพผลลัพธ์", "- วิธีพล็อตผลลัพธ์"),
    ("ขั้นแรก นำเข้าโมดูลที่จำเป็นและตั้งค่าสภาพแวดล้อมของเราและโหลดโมดูลที่จำเป็น\nโมดูล", "เริ่มต้นด้วยการนำเข้าโมดูลที่จำเป็นและตั้งค่าสภาพแวดล้อมสำหรับการทำงาน"),
    ("HENS มี checkpoint เก็บไว้อย่างสะดวกบน", "HENS มี checkpoint ให้ใช้งานอยู่บน"),
    ("ที่เราจะใช้", "ซึ่งเราจะใช้ในตัวอย่างนี้"),
    ("แทนที่จะโหลด checkpoint เริ่มต้นจากกระดาษ SFNO ต้นฉบับ ให้สร้างไฟล์\nmodel package ที่ชี้ไปที่ HENS checkpoint เฉพาะที่เราต้องการใช้แทน", "แทนที่จะโหลด checkpoint เริ่มต้นจากงาน SFNO ต้นฉบับ เราจะสร้าง model package ที่ชี้ไปยัง HENS checkpoint เฉพาะที่ต้องการใช้งาน"),
    ("ตัวอย่างนี้ยังต้องการสิ่งต่อไปนี้:", "ตัวอย่างนี้ยังต้องใช้องค์ประกอบต่อไปนี้:"),
    ("- โมเดลพยากรณ์พื้นฐาน: ใช้สถาปัตยกรรมโมเดล SFNO", "- Prognostic Model พื้นฐาน: ใช้สถาปัตยกรรมโมเดล SFNO"),
    ("- วิธีการก่อกวน: HENS ใช้วิธีการก่อกวนแบบใหม่", "- Perturbation Method: HENS ใช้วิธี perturbation แบบใหม่"),
    ("- วิธีการ seed: วิธีการเพาะเมล็ดพันธุ์ Vector", "- Seeding Method: ใช้วิธี Correlated Spherical Gaussian"),
]


EXACT_CELL_REPLACEMENTS = {
    "\n# การรันการอนุมานเชิงกำหนด\n\nworkflow พื้นฐานสำหรับ deterministic inference\n\nตัวอย่างนี้จะสาธิตวิธีการรัน inference workflow อย่างง่ายเพื่อสร้าง\ndeterministic forecast เบื้องต้นโดยใช้หนึ่งในโมเดลในตัวของการอนุมาน Earth-2\nStudio\n\nในตัวอย่างนี้คุณจะได้เรียนรู้:\n\n- วิธีสร้างอินสแตนซ์ของโมเดลพยากรณ์ที่มีมาให้ในระบบ\n- วิธีสร้างแหล่งข้อมูลและออบเจ็กต์ IO\n- วิธีรัน workflow พื้นฐานที่มีมาให้ในระบบ\n- วิธีทำ post-processing กับผลลัพธ์\n": "\n# การรัน Deterministic Inference\n\nworkflow พื้นฐานสำหรับ deterministic inference\n\nตัวอย่างนี้จะแสดงวิธีรัน inference workflow แบบง่าย เพื่อสร้าง deterministic forecast เบื้องต้น โดยใช้หนึ่งในโมเดลที่มีมาให้ภายใน Earth-2 Inference\nStudio\n\nในตัวอย่างนี้คุณจะได้เรียนรู้:\n\n- วิธีสร้างอินสแตนซ์ของโมเดลพยากรณ์ที่มีมาให้ในระบบ\n- วิธีสร้างแหล่งข้อมูลและออบเจ็กต์ IO\n- วิธีรัน workflow พื้นฐานที่มีมาให้ในระบบ\n- วิธีทำ post-processing กับผลลัพธ์\n",
    "\n# กำลังเรียกใช้การอนุมานการวินิจฉัย\n\nการพยากรณ์ขั้นพื้นฐาน + การวินิจฉัย inference workflow\n\nตัวอย่างนี้จะสาธิตวิธีการรัน deterministic inference workflow ที่จับคู่กัน\nPrognostic Model พร้อมด้วย Diagnostic Model Diagnostic Model นี้จะทำนายใหม่\nปริมาณบรรยากาศจากสาขาพยากรณ์ของการพยากรณ์\n\nในตัวอย่างนี้คุณจะได้เรียนรู้:\n\n- วิธีสร้างอินสแตนซ์ Prognostic Model\n- วิธีสร้างอินสแตนซ์ Diagnostic Model\n- วิธีสร้างแหล่งข้อมูลและออบเจ็กต์ IO\n- เรียกใช้การวินิจฉัย workflow ในตัว\n- วิธีทำ post-processing กับผลลัพธ์\n": "\n# การรัน Diagnostic Inference\n\nworkflow พื้นฐานสำหรับการใช้โมเดลพยากรณ์ร่วมกับโมเดลวินิจฉัย\n\nตัวอย่างนี้จะแสดงวิธีรัน deterministic inference workflow ที่ใช้ Prognostic Model ควบคู่กับ Diagnostic Model โดยโมเดลวินิจฉัยจะทำนายตัวแปรบรรยากาศเพิ่มเติมจากฟิลด์พยากรณ์ที่ได้จากโมเดลพยากรณ์\n\nในตัวอย่างนี้คุณจะได้เรียนรู้:\n\n- วิธีสร้างอินสแตนซ์ของ Prognostic Model\n- วิธีสร้างอินสแตนซ์ของ Diagnostic Model\n- วิธีสร้างแหล่งข้อมูลและออบเจ็กต์ IO\n- วิธีรัน diagnostic workflow ที่มีมาให้ในระบบ\n- วิธีทำ post-processing กับผลลัพธ์\n",
    "\n# ขยาย Prognostic Model\n\nการใช้งาน Prognostic Model แบบกำหนดเอง\n\nตัวอย่างนี้จะสาธิตวิธีการขยาย Earth2Studio โดยการนำแบบกำหนดเองไปใช้\nPrognostic Model และรันใน workflow ทั่วไป\n\nในตัวอย่างนี้คุณจะได้เรียนรู้:\n\n- ข้อกำหนด API ของแบบจำลองการพยากรณ์\n- การใช้ Prognostic Model แบบกำหนดเอง\n- การรันโมเดลนี้ใน workflow ที่มีอยู่\n": "\n# การขยาย Prognostic Model\n\nการใช้งาน Prognostic Model แบบกำหนดเอง\n\nตัวอย่างนี้จะแสดงวิธีขยาย Earth2Studio โดยการสร้าง Prognostic Model แบบกำหนดเอง แล้วนำไปรันใน workflow ทั่วไปของระบบ\n\nในตัวอย่างนี้คุณจะได้เรียนรู้:\n\n- ข้อกำหนดของ API สำหรับ Prognostic Model\n- วิธีสร้าง Prognostic Model แบบกำหนดเอง\n- วิธีนำโมเดลนี้ไปรันใน workflow ที่มีอยู่\n",
    "## การพยากรณ์แบบกำหนดเอง\nตามที่กล่าวไว้ในส่วน `prognostic_model_userguide` ของคู่มือผู้ใช้\nEarth2Studio กำหนด Prognostic Model ผ่านอินเทอร์เฟซที่เรียบง่าย\n:py:class:`earth2studio.models.px.base.PrognosticModel`. สามารถใช้เพื่อช่วยได้\nแนะนำ API ที่จำเป็นซึ่งจำเป็นในการสร้างการพยากรณ์แบบกำหนดเองของเราเองได้สำเร็จ\n\nในตัวอย่างนี้ เรามาสร้างการพยากรณ์แบบง่ายๆ ที่คาดการณ์ว่าการเพิ่ม noise แบบ Gaussian\nเข้าไปในสนามลมผิวพื้นในทุก step แม้ว่าจะไม่สามารถใช้งานได้จริง แต่ก็ควรทำเช่นนี้\nสาธิต API ที่เราจำเป็นต้องนำไปใช้สำหรับการพยากรณ์\n\nเริ่มต้นด้วยตัวสร้าง โมเดลการพยากรณ์ควรเป็นโมดูล PyTorch\nโมเดลจำเป็นต้องมีวิธี :py:obj:`to(device)` ที่สามารถย้ายโมเดลระหว่างกันได้\nอุปกรณ์ที่แตกต่างกัน หากโมเดลของคุณคือ PyTorch สิ่งนี้จะเป็นเรื่องง่าย\n\n": "## Prognostic Model แบบกำหนดเอง\nตามที่อธิบายไว้ใน `prognostic_model_userguide` ของคู่มือผู้ใช้ Earth2Studio กำหนดให้ Prognostic Model เป็นอินเทอร์เฟซที่เรียบง่ายผ่าน\n:py:class:`earth2studio.models.px.base.PrognosticModel` ซึ่งช่วยกำหนด API ที่จำเป็นสำหรับการสร้างโมเดลพยากรณ์แบบกำหนดเองของเราเอง\n\nในตัวอย่างนี้ เราจะสร้างโมเดลอย่างง่ายที่เพิ่ม Gaussian noise เข้าไปในสนามลมผิวพื้นในทุก step แม้จะไม่ใช่โมเดลที่ใช้จริง แต่เหมาะสำหรับสาธิต API ที่จำเป็นต้องมีสำหรับ Prognostic Model\n\nเริ่มจาก constructor ก่อน โดยทั่วไปโมเดลพยากรณ์ควรเป็นโมดูล PyTorch และควรมีเมธอด :py:obj:`to(device)` เพื่อย้ายโมเดลข้ามอุปกรณ์ต่าง ๆ ได้ หากโมเดลของคุณเขียนด้วย PyTorch อยู่แล้ว ส่วนนี้จะทำได้ไม่ยาก\n\n",
    "\n# จุดตรวจตระการตาขนาดใหญ่ (HENS)\n\ncheckpoint Huge Ensembles (HENS) หลายตัวพื้นฐาน inference workflow\n\nตัวอย่างนี้เป็นตัวอย่างพื้นฐานในการโหลด Huge Ensemble checkpoints เพื่อดำเนินการ\nensemble inference.\nโน้ตบุ๊คนี้มีจุดมุ่งหมายเพื่อแสดงพื้นฐานของการใช้งาน multi-checkpoint workflow\nจากส่วนประกอบ Earth2Studio\nสำหรับรายละเอียดเพิ่มเติมเกี่ยวกับ HENS โปรดดู:\n\n- https://arxiv.org/abs/2408.03100\n- https://github.com/ankurmahesh/earth2mip-fork\n\n\n<div class=\"alert alert-danger\"><h4>คำเตือน</h4><p>เราขอแนะนำให้ผู้ใช้ทำความคุ้นเคยกับข้อจำกัดใบอนุญาตของสิ่งนี้\ncheckpoint ของโมเดล</p></div>\n\nสำหรับ HENS workflow ที่สมบูรณ์ เราขอแนะนำให้ผู้ใช้ดูสูตร HENS\nซึ่งนำเสนอโซลูชั่นแบบ end-to-end เพื่อใช้ประโยชน์จาก HENS สำหรับการวิเคราะห์ขั้นปลาย เช่น\nการติดตามพายุหมุนเขตร้อน:\n\n- https://github.com/NVIDIA/earth2studio/tree/main/recipes/hens\n\nในตัวอย่างนี้คุณจะได้เรียนรู้:\n\n- วิธีโหลด HENS checkpoints ด้วยแพ็คเกจโมเดลแบบกำหนดเอง\n- วิธีโหลดวิธีก่อกวน HENS\n- วิธีสร้างลูป ensemble inference อย่างง่าย\n- วิธีการเห็นภาพผลลัพธ์\n": "\n# Huge Ensembles (HENS)\n\nworkflow พื้นฐานสำหรับการใช้งาน Huge Ensemble checkpoints หลายชุด\n\nตัวอย่างนี้แสดงวิธีโหลด Huge Ensemble checkpoints เพื่อรัน ensemble inference โดยมีจุดมุ่งหมายเพื่อสาธิตพื้นฐานของ multi-checkpoint workflow จากคอมโพเนนต์ของ Earth2Studio\n\nสำหรับรายละเอียดเพิ่มเติมเกี่ยวกับ HENS โปรดดู:\n\n- https://arxiv.org/abs/2408.03100\n- https://github.com/ankurmahesh/earth2mip-fork\n\n<div class=\"alert alert-danger\"><h4>คำเตือน</h4><p>ขอแนะนำให้ผู้ใช้ตรวจสอบข้อจำกัดด้านใบอนุญาตของ checkpoint โมเดลนี้ก่อนใช้งาน</p></div>\n\nสำหรับ HENS workflow แบบครบถ้วน เราแนะนำให้ดู recipe ของ HENS ซึ่งเป็นโซลูชันแบบ end-to-end สำหรับนำ HENS ไปใช้กับงานวิเคราะห์ขั้นปลาย เช่น การติดตามพายุหมุนเขตร้อน:\n\n- https://github.com/NVIDIA/earth2studio/tree/main/recipes/hens\n\nในตัวอย่างนี้คุณจะได้เรียนรู้:\n\n- วิธีโหลด HENS checkpoints ด้วย model package แบบกำหนดเอง\n- วิธีโหลดวิธี perturbation ของ HENS\n- วิธีสร้างลูป ensemble inference แบบง่าย\n- วิธีพล็อตผลลัพธ์\n",
}


def polish_text(text: str) -> str:
    if text in EXACT_CELL_REPLACEMENTS:
        return EXACT_CELL_REPLACEMENTS[text]
    out = text
    for old, new in GLOBAL_REPLACEMENTS:
        out = out.replace(old, new)
    return out


def main() -> int:
    for path in sorted(Path(".").glob("**/*.th.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for cell in nb.get("cells", []):
            source = "".join(cell.get("source", []))
            polished = polish_text(source)
            if polished != source:
                cell["source"] = [polished]
                changed = True
        if changed:
            path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
