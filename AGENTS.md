# CRISM Journal — Agent Notes

## Mineral identification (repeated task)

Use the project skill `.cursor/skills/crism-mineral-identification/SKILL.md` and rule `.cursor/rules/crism-mineral-pipeline.mdc`.

**Full-height FRT scenes (preferred):**

```powershell
cd "D:\CRISM Journal\cv-crism"
$env:PYTHONIOENCODING = "utf-8"
python run_mineral_fullfrt.py --scene 1
```

Scenes live at `D:\CRISM scenes\scenes\Scene N\` (`.img` + `.lbl` co-located).

**Legacy subsets:**

```powershell
$env:CRISM_DATA_DIR = "D:\NEW CROSS MISSION\Data\Scene 12\Crism"
python Stage5_minerals.py
```

See `MINERAL_ID_FIXES.md` for the issue→fix map.
