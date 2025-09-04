"""
PyQt6 Node-based Data Preprocessor (fixed + headless tests)
-----------------------------------------------------------
- Fix: use QPainter.RenderHint.Antialiasing correctly.
- Fix: painter.drawText overloads (use QPointF/QRectF and cast flags to int).
- Fix: source node returns cached DataFrame (e.g., "Load CSV").
- Add: headless CLI and tests for core pandas ops.

Usage
-----
GUI (needs PyQt6):
    python node_preprocessor.py

Headless demo:
    python node_preprocessor.py --csv path/to.csv --ops dropna head5 describe

Run tests:
    python node_preprocessor.py --run-tests

Install:
    pip install pyqt6 pandas
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Dict, List, Optional

# ------------------------------
# Optional imports
# ------------------------------

try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

try:  # GUI is optional
    from PyQt6.QtCore import Qt, QPointF, QRectF
    from PyQt6.QtGui import QPainter, QPainterPath, QPen, QBrush, QAction
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene, QGraphicsItem,
        QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPathItem, QLabel, QPushButton,
        QFileDialog, QMessageBox
    )
    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False

# ------------------------------
# Headless core operations
# ------------------------------

def op_dropna(df):
    if df is None:
        return None
    return df.dropna()

def op_head5(df):
    if df is None:
        return None
    return df.head(5)

def op_describe(df):
    if df is None:
        return None
    return df.describe()

OP_REGISTRY: Dict[str, Callable[[Any], Any]] = {
    "dropna": op_dropna,
    "drop_na": op_dropna,
    "drop-na": op_dropna,
    "head5": op_head5,
    "head(5)": op_head5,
    "describe": op_describe,
}


def run_headless_pipeline(csv_path: Optional[str], ops: List[str]) -> Any:
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required for headless pipeline; please install pandas")
    df = None
    if csv_path:
        df = pd.read_csv(csv_path)
    for op in ops:
        fn = OP_REGISTRY.get(op.lower())
        if fn is None:
            raise ValueError(f"Unknown op: {op}. Available: {sorted(OP_REGISTRY)}")
        df = fn(df)
    return df


def run_tests() -> None:
    if not PANDAS_AVAILABLE:
        print("SKIP tests: pandas not available")
        return
    # Test data
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5, None],
        "b": [10, None, 30, 40, 50, 60],
    })
    # 1) dropna should remove rows with any NaNs
    out1 = op_dropna(df)
    assert out1.shape[0] == 3, f"dropna rows expected 3, got {out1.shape[0]}"
    # 2) head5 should cap rows to 5
    out2 = op_head5(df)
    assert out2.shape[0] == 5, f"head5 rows expected 5, got {out2.shape[0]}"
    # 3) describe should return stats with index containing 'mean' for numeric
    out3 = op_describe(df)
    assert "mean" in out3.index, "describe expected 'mean' row"
    # 4) composition with data: dropna then head5 -> at most 3 rows
    out4 = op_head5(op_dropna(df))
    assert out4.shape[0] == 3, f"chained rows expected 3, got {out4.shape[0]}"
    # 5) pipeline without CSV should yield None and not crash
    out5 = run_headless_pipeline(None, ["dropna", "head5", "describe"]) if PANDAS_AVAILABLE else None
    assert out5 is None, "Pipeline without data should yield None"
    print("All tests passed ✔")


# ------------------------------
# GUI implementation (only if PyQt6 is present)
# ------------------------------

if PYQT_AVAILABLE:
    class Port(QGraphicsEllipseItem):
        """A small circular port for connecting nodes."""
        RADIUS = 6

        def __init__(self, parent: QGraphicsItem, is_output: bool, key: str):
            super().__init__(-Port.RADIUS, -Port.RADIUS, 2*Port.RADIUS, 2*Port.RADIUS, parent)
            self.setBrush(QBrush(Qt.GlobalColor.white))
            self.setPen(QPen(Qt.GlobalColor.black, 1.2))
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
            self.is_output = is_output
            self.key = key
            self.connection: Optional[Connection] = None  # single-connection per-port for simplicity

        def can_connect(self, other: 'Port') -> bool:
            return (
                other is not None
                and other is not self
                and self.is_output != other.is_output
                and self.connection is None
                and other.connection is None
            )


    class Connection(QGraphicsPathItem):
        """Curved wire connecting two ports."""
        def __init__(self, a: Port, b: Optional[Port] = None):
            super().__init__()
            self.a = a
            self.b = b
            self.temp_end = a.scenePos()
            self.setZValue(-1)  # draw behind nodes
            self.setPen(QPen(Qt.GlobalColor.black, 2))
            self.update_path()

        def set_temp_end(self, pos: QPointF):
            self.temp_end = pos
            self.update_path()

        def update_path(self):
            p1 = self.a.scenePos()
            p2 = self.b.scenePos() if self.b else self.temp_end
            dx = max(40, abs(p2.x() - p1.x()) * 0.5)
            c1 = QPointF(p1.x() + dx, p1.y())
            c2 = QPointF(p2.x() - dx, p2.y())
            path = QPainterPath(p1)
            path.cubicTo(c1, c2, p2)
            self.setPath(path)


    class NodeItem(QGraphicsRectItem):
        WIDTH = 180
        HEADER_H = 28

        def __init__(self, title: str, fn: Callable[[Any], Any] | None, inputs: List[str], outputs: List[str]):
            super().__init__(0, 0, NodeItem.WIDTH, NodeItem.HEADER_H + 24 + max(len(inputs), len(outputs)) * 20)
            self.setBrush(QBrush(Qt.GlobalColor.lightGray))
            self.setPen(QPen(Qt.GlobalColor.black, 1.5))
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            self.title = title
            self.fn = fn
            self.inputs: Dict[str, Port] = {}
            self.outputs: Dict[str, Port] = {}
            self.value_cache: Any = None

            # header
            self.header = QGraphicsRectItem(0, 0, NodeItem.WIDTH, NodeItem.HEADER_H, self)
            self.header.setBrush(QBrush(Qt.GlobalColor.darkGray))
            self.header.setPen(QPen(Qt.GlobalColor.black, 0))

            # ports
            for i, key in enumerate(inputs):
                port = Port(self, is_output=False, key=key)
                port.setPos(8, NodeItem.HEADER_H + 20 + i * 20)
                self.inputs[key] = port
            for i, key in enumerate(outputs):
                port = Port(self, is_output=True, key=key)
                port.setPos(self.rect().width() - 8, NodeItem.HEADER_H + 20 + i * 20)
                self.outputs[key] = port

        def boundingRect(self) -> QRectF:
            return super().boundingRect()

        def paint(self, painter, option, widget=None):
            super().paint(painter, option, widget)
            # Title text in header using QRectF overload
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(
                QRectF(8.0, 0.0, float(self.rect().width()-16.0), float(NodeItem.HEADER_H)),
                int(Qt.AlignmentFlag.AlignVCenter),
                self.title,
            )
            painter.setPen(Qt.GlobalColor.black)
            # Port labels using QPointF overload to avoid float/int mismatch
            y0 = NodeItem.HEADER_H + 20
            for i, key in enumerate(self.inputs.keys()):
                painter.drawText(QPointF(18.0, float(y0 + i*20 + 4)), key)
            for i, key in enumerate(self.outputs.keys()):
                x = float(self.rect().width() - 100.0)
                painter.drawText(QPointF(x, float(y0 + i*20 + 4)), key)

        def evaluate(self) -> Any:
            """Evaluate this node.
            - If it has inputs: pull upstream value and apply self.fn (or passthrough).
            - If it has NO inputs (source node like Load CSV): return cached value.
            """
            if not self.inputs:  # source node
                return self.value_cache

            inbound: Optional[Port] = next(iter(self.inputs.values()), None)
            upstream_val = None
            if inbound and inbound.connection:
                upstream_node = inbound.connection.a.parentItem()  # type: ignore
                if isinstance(upstream_node, NodeItem):
                    upstream_val = upstream_node.evaluate()
            if self.fn:
                try:
                    self.value_cache = self.fn(upstream_val)
                except Exception as e:
                    self.value_cache = f"Error: {e}"
            else:
                self.value_cache = upstream_val
            return self.value_cache


    class NodeScene(QGraphicsScene):
        def __init__(self):
            super().__init__()
            self.temp_connection: Optional[Connection] = None
            self.grab_port: Optional[Port] = None

        def start_connection(self, port: Port):
            self.grab_port = port
            if port.is_output:
                self.temp_connection = Connection(port)
            else:
                # start from input: allow dragging backwards
                dummy_out = Port(port.parentItem(), True, "_tmp")
                dummy_out.setPos(port.scenePos())
                self.temp_connection = Connection(dummy_out)
            self.addItem(self.temp_connection)

        def end_connection(self, port: Optional[Port]):
            if not self.temp_connection:
                return
            a = self.grab_port if self.grab_port and self.grab_port.is_output else None
            b = port if port and not port.is_output else None
            # Swap if user dragged from input to output
            if a is None and port and port.is_output and isinstance(self.grab_port, Port) and not self.grab_port.is_output:
                a = port
                b = self.grab_port
            if a and b and a.can_connect(b):
                self.temp_connection.a = a
                self.temp_connection.b = b
                self.temp_connection.update_path()
                a.connection = self.temp_connection
                b.connection = self.temp_connection
            else:
                self.removeItem(self.temp_connection)
            self.temp_connection = None
            self.grab_port = None

        def mouseMoveEvent(self, event):
            if self.temp_connection:
                self.temp_connection.set_temp_end(event.scenePos())
            super().mouseMoveEvent(event)


    class View(QGraphicsView):
        def __init__(self, scene: NodeScene):
            super().__init__(scene)
            # Use QPainter.RenderHint
            self.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

        def mousePressEvent(self, event):
            item = self.itemAt(event.pos())
            port = item if isinstance(item, Port) else None
            if port:
                self.scene().start_connection(port)  # type: ignore
            super().mousePressEvent(event)

        def mouseReleaseEvent(self, event):
            item = self.itemAt(event.pos())
            port = item if isinstance(item, Port) else None
            self.scene().end_connection(port)  # type: ignore
            super().mouseReleaseEvent(event)


    def node_definitions() -> Dict[str, Dict]:
        return {
            "Load CSV": {
                "inputs": [],
                "outputs": ["df"],
                "fn": None,  # handled specially via file picker
            },
            "Drop NA": {
                "inputs": ["df"],
                "outputs": ["df"],
                "fn": op_dropna,
            },
            "Head(5)": {
                "inputs": ["df"],
                "outputs": ["df"],
                "fn": op_head5,
            },
            "Describe": {
                "inputs": ["df"],
                "outputs": ["df"],
                "fn": op_describe,
            },
        }


    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Node Preprocessor (PyQt6 minimal)")
            self.resize(1000, 640)

            central = QWidget()
            self.setCentralWidget(central)
            layout = QHBoxLayout(central)

            # Palette
            left = QVBoxLayout()
            self.palette = QListWidget()
            for name in node_definitions().keys():
                QListWidgetItem(name, self.palette)
            self.palette.itemDoubleClicked.connect(self.add_node_from_palette)
            left.addWidget(QLabel("Blocks"))
            left.addWidget(self.palette)
            self.eval_btn = QPushButton("Run pipeline from selected -> print")
            self.eval_btn.clicked.connect(self.evaluate_from_selection)
            left.addWidget(self.eval_btn)

            # Graphics view
            self.scene = NodeScene()
            self.view = View(self.scene)

            layout.addLayout(left, 0)
            layout.addWidget(self.view, 1)

            # Context action: set CSV path on a Load CSV node
            self.load_csv_action = QAction("Load CSV...", self)
            self.load_csv_action.triggered.connect(self.configure_load_csv)
            self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
            self.view.addAction(self.load_csv_action)

            # Storage for node-specific state (e.g., file paths)
            self.node_state: Dict[NodeItem, Dict[str, Any]] = {}

        def add_node_from_palette(self, item: QListWidgetItem):
            name = item.text()
            spec = node_definitions()[name]
            node = NodeItem(name, spec.get("fn"), spec.get("inputs", []), spec.get("outputs", []))
            node.setPos(self.view.mapToScene(260, 120))
            self.scene.addItem(node)
            if name == "Load CSV":
                self.node_state[node] = {"path": None}

        def configure_load_csv(self):
            # Find a selected Load CSV node
            selected = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem) and i.title == "Load CSV"]
            if not selected:
                QMessageBox.information(self, "Select node", "Select a 'Load CSV' node first.")
                return
            node: NodeItem = selected[0]
            path, _ = QFileDialog.getOpenFileName(self, "Choose CSV", filter="CSV Files (*.csv)")
            if not path:
                return
            try:
                df = pd.read_csv(path)
            except Exception as e:
                QMessageBox.critical(self, "Read error", str(e))
                return
            self.node_state[node] = {"path": path, "data": df}
            node.value_cache = df  # cache on the source node
            msg = _format_loaded_message(len(df), path)
            QMessageBox.information(self, "Loaded", msg)} rows from
{path}")

        def evaluate_from_selection(self):
            selected = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
            if not selected:
                QMessageBox.information(self, "Select", "Select a node to evaluate from.")
                return
            start: NodeItem = selected[0]
            result = start.evaluate()
            # pretty print result
            try:
                if PANDAS_AVAILABLE and isinstance(result, pd.DataFrame):
                    info = f"DataFrame shape: {result.shape}
Columns: {list(result.columns)[:8]}..."
                else:
                    info = str(result)
            except Exception:
                info = str(result)
            print("
===== PIPELINE RESULT =====
", info)
            QMessageBox.information(self, "Result", info[:1000])


# ------------------------------
# Entrypoint
# ------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV for headless ops")
    parser.add_argument("--ops", nargs='*', default=[], help="Sequence of ops: dropna head5 describe")
    parser.add_argument("--run-tests", action="store_true", help="Run headless tests and exit")
    parser.add_argument("--gui", action="store_true", help="Force GUI (requires PyQt6)")
    args = parser.parse_args(argv)

    if args.run_tests:
        run_tests()
        return 0

    # Headless path if ops or csv specified (useful when PyQt6 not installed)
    if args.csv or args.ops:
        res = run_headless_pipeline(args.csv, args.ops)
        if PANDAS_AVAILABLE and isinstance(res, pd.DataFrame):
            print("Headless result:", res.shape)
            print(res.head(10))
        else:
            print("Headless result:", res)
        return 0

    # Otherwise, try GUI
    if not PYQT_AVAILABLE and args.gui:
        print("PyQt6 not available. Install with: pip install pyqt6")
        return 1

    if PYQT_AVAILABLE:
        app = QApplication(sys.argv)
        win = MainWindow()
        win.show()
        return app.exec()

    # Fallback when no GUI and no headless args
    print("PyQt6 not installed. Run with --run-tests or provide --csv/--ops for headless mode.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
