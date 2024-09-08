import React, { useEffect, useRef } from "react";
import * as NGL from "ngl";

function NglViewer({ pdbUrl }) {
  const viewerRef = useRef(null);

  useEffect(() => {
    const stage = new NGL.Stage("ngl-viewer");
    stage.loadFile(pdbUrl).then((component) => {
      component.addRepresentation("cartoon");
      component.autoView();
    });

    return () => {
      stage.dispose();
    };
  }, [pdbUrl]);

  return (
    <div
      id="ngl-viewer"
      ref={viewerRef}
      style={{ width: "100%", height: "400px" }}
    />
  );
}

export default NglViewer;
