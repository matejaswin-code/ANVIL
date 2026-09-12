"""ANVIL Blender addon.

Runs the same anvil_core pipeline, but embedded — so blender_bridge takes its
in-process branch and scripts execute against the live scene rather than through
a `blender --background` subprocess.

Install: Edit > Preferences > Add-ons > Install…, point at this folder zipped, or
symlink it into Blender's addons directory. The addon needs anvil_core on the
path; set ANVIL_PROJECT_ROOT if the addon lives outside the project directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

bl_info = {
    "name": "ANVIL — AI 3D asset pipeline",
    "author": "ANVIL",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > ANVIL",
    "description": "Local-first text/image → 3D model pipeline, straight into the scene",
    "category": "Object",
}

import bpy  # noqa: E402


def _ensure_core_on_path() -> None:
    root = os.environ.get("ANVIL_PROJECT_ROOT")
    candidates = [Path(root)] if root else []
    candidates.append(Path(__file__).resolve().parent.parent)
    for candidate in candidates:
        if (candidate / "anvil_core").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


_ensure_core_on_path()

try:
    from anvil_core import gpu_check, schemas, sessions, state_machine, styles
    from anvil_core.config import SETTINGS
    CORE_AVAILABLE = True
    CORE_ERROR = ""
except Exception as exc:  # noqa: BLE001
    CORE_AVAILABLE = False
    CORE_ERROR = str(exc)


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------

def _style_items(self, context):
    if not CORE_AVAILABLE:
        return [("realistic", "Realistic", "")]
    return [(s.id, s.label, s.notes or s.label) for s in styles.STYLE_PRESETS]


class AnvilProperties(bpy.types.PropertyGroup):
    mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ("text_to_image", "Text → Image", "Mandatory checklist, generates a concept image first"),
            ("text", "Direct text", "All fields optional"),
            ("image_upload", "Image upload", "Generation flags, no prompt text"),
        ],
        default="text_to_image",
    )
    style_id: bpy.props.EnumProperty(name="Style", items=_style_items)
    procedural: bpy.props.BoolProperty(
        name="Procedural mode",
        description="Skip neural mesh-gen — the LLM writes bpy construction code directly",
        default=False,
    )
    object_type: bpy.props.StringProperty(name="Object type", default="")
    style_desc: bpy.props.StringProperty(name="Style descriptor", default="")
    primary_material: bpy.props.StringProperty(name="Primary material", default="")
    color_palette: bpy.props.StringProperty(name="Color palette", default="")
    image_path: bpy.props.StringProperty(name="Reference image", subtype="FILE_PATH", default="")
    edit_instruction: bpy.props.StringProperty(name="Edit", default="")
    status: bpy.props.StringProperty(name="Status", default="Ready")
    last_session: bpy.props.StringProperty(default="")


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

class ANVIL_OT_generate(bpy.types.Operator):
    bl_idname = "anvil.generate"
    bl_label = "Generate"
    bl_description = "Run the full generation pipeline and import the result into this scene"

    def execute(self, context):
        if not CORE_AVAILABLE:
            self.report({"ERROR"}, f"anvil_core unavailable: {CORE_ERROR}")
            return {"CANCELLED"}

        props = context.scene.anvil
        checklist = {
            "object_type": props.object_type,
            "style_desc": props.style_desc,
            "primary_material": props.primary_material,
            "color_palette": props.color_palette,
        }
        checklist = {k: v for k, v in checklist.items() if v.strip()}

        missing = schemas.missing_mandatory(props.mode, checklist)
        if missing:
            self.report({"ERROR"}, f"Fill in the required fields first: {', '.join(missing)}")
            return {"CANCELLED"}

        session = schemas.Session(
            mode=props.mode,
            style_id=props.style_id,
            procedural=props.procedural,
            checklist=checklist,
            uploaded_image_path=bpy.path.abspath(props.image_path) if props.image_path else None,
        )
        sessions.save_session(session)
        props.status = "Generating…"

        try:
            # Blocking on purpose: Blender's UI isn't thread-safe for scene edits,
            # and the import step must run on the main thread anyway.
            state_machine.run_generation_pipeline(session)
        except Exception as exc:  # noqa: BLE001
            props.status = f"Failed: {exc}"
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        props.last_session = session.session_id
        props.status = f"Done — session {session.session_id[:8]}"
        self.report({"INFO"}, "Generation complete")
        return {"FINISHED"}


class ANVIL_OT_edit(bpy.types.Operator):
    bl_idname = "anvil.edit"
    bl_label = "Send edit"

    def execute(self, context):
        if not CORE_AVAILABLE:
            self.report({"ERROR"}, f"anvil_core unavailable: {CORE_ERROR}")
            return {"CANCELLED"}
        from anvil_core import edit_loop

        props = context.scene.anvil
        session = sessions.load_session(props.last_session)
        if session is None:
            self.report({"ERROR"}, "No finished session to edit. Generate something first.")
            return {"CANCELLED"}
        try:
            result = edit_loop.handle_edit(session, props.edit_instruction, confirmed=True)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        props.edit_instruction = ""
        props.status = result.get("message", "Edit applied")
        self.report({"INFO"}, props.status)
        return {"FINISHED"}


class ANVIL_OT_undo(bpy.types.Operator):
    bl_idname = "anvil.undo_edit"
    bl_label = "Undo last edit"

    def execute(self, context):
        from anvil_core import edit_loop
        props = context.scene.anvil
        session = sessions.load_session(props.last_session)
        if session is None:
            self.report({"WARNING"}, "No session loaded")
            return {"CANCELLED"}
        result = edit_loop.undo(session)
        props.status = result["message"]
        self.report({"INFO"} if result["ok"] else {"WARNING"}, result["message"])
        return {"FINISHED"}


class ANVIL_OT_rig(bpy.types.Operator):
    bl_idname = "anvil.rig"
    bl_label = "Rig this"

    def execute(self, context):
        from anvil_core import edit_loop
        props = context.scene.anvil
        session = sessions.load_session(props.last_session)
        if session is None:
            self.report({"ERROR"}, "No session to rig")
            return {"CANCELLED"}
        try:
            result = edit_loop.handle_edit(session, "[Rig this]", confirmed=True)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        props.status = result.get("message", "Rigged")
        return {"FINISHED"}


class ANVIL_OT_open_ui(bpy.types.Operator):
    bl_idname = "anvil.open_web_ui"
    bl_label = "Open the full workbench"
    bl_description = "Open the browser UI (start the server separately with run.sh / run.bat)"

    def execute(self, context):
        import webbrowser
        webbrowser.open(f"http://{SETTINGS.host}:{SETTINGS.port}/")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------

class ANVIL_PT_panel(bpy.types.Panel):
    bl_label = "ANVIL"
    bl_idname = "ANVIL_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ANVIL"

    def draw(self, context):
        layout = self.layout
        props = context.scene.anvil

        if not CORE_AVAILABLE:
            box = layout.box()
            box.label(text="anvil_core not importable", icon="ERROR")
            box.label(text=CORE_ERROR[:60])
            box.label(text="Set ANVIL_PROJECT_ROOT and restart Blender.")
            return

        cap = gpu_check.check_gpu_capability()
        if cap.get("warning"):
            box = layout.box()
            box.label(text="Capability", icon="ERROR" if cap["blocked"] else "INFO")
            for chunk in _wrap(cap["warning"], 42):
                box.label(text=chunk)

        layout.prop(props, "mode", text="")
        layout.prop(props, "style_id", text="")
        layout.prop(props, "procedural")

        col = layout.column(align=True)
        if props.mode == "image_upload":
            col.prop(props, "image_path", text="")
        else:
            required = props.mode == "text_to_image"
            col.prop(props, "object_type", text="Object" + ("*" if required else ""))
            col.prop(props, "style_desc", text="Style" + ("*" if required else ""))
            col.prop(props, "primary_material", text="Material" + ("*" if required else ""))
            col.prop(props, "color_palette", text="Palette" + ("*" if required else ""))

        layout.separator()
        layout.operator("anvil.generate", icon="SHADERFX")

        if props.last_session:
            layout.separator()
            box = layout.box()
            box.label(text="Edit", icon="GREASEPENCIL")
            box.prop(props, "edit_instruction", text="")
            row = box.row(align=True)
            row.operator("anvil.edit", text="Send")
            row.operator("anvil.undo_edit", text="Undo")
            box.operator("anvil.rig", icon="ARMATURE_DATA")

        layout.separator()
        layout.operator("anvil.open_web_ui", icon="WORLD")
        layout.label(text=props.status[:48], icon="INFO")


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width) or [""]


CLASSES = (
    AnvilProperties,
    ANVIL_OT_generate,
    ANVIL_OT_edit,
    ANVIL_OT_undo,
    ANVIL_OT_rig,
    ANVIL_OT_open_ui,
    ANVIL_PT_panel,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.anvil = bpy.props.PointerProperty(type=AnvilProperties)
    # §2.11 — capability check runs once at addon registration, not per session.
    if CORE_AVAILABLE:
        gpu_check.check_gpu_capability()


def unregister() -> None:
    del bpy.types.Scene.anvil
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
