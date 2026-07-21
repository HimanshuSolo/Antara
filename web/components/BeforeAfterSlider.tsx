"use client";

import { useCallback, useRef, useState } from "react";

export default function BeforeAfterSlider({
  leftSrc,
  rightSrc,
  leftLabel,
  rightLabel,
  alt,
}: {
  leftSrc: string;
  rightSrc: string;
  leftLabel: string;
  rightLabel: string;
  alt: string;
}) {
  const [position, setPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updateFromClientX = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPosition(Math.min(100, Math.max(0, pct)));
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    dragging.current = true;
    (e.target as Element).setPointerCapture(e.pointerId);
    updateFromClientX(e.clientX);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging.current) return;
    updateFromClientX(e.clientX);
  };

  const onPointerUp = () => {
    dragging.current = false;
  };

  return (
    <div>
      <div
        className="slider"
        ref={containerRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={rightSrc} alt={`${alt} (${rightLabel})`} />
        <div className="slider__clip" style={{ width: `${position}%` }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={leftSrc} alt={`${alt} (${leftLabel})`} />
        </div>
        <div className="slider__tag slider__tag--left">{leftLabel}</div>
        <div className="slider__tag slider__tag--right">{rightLabel}</div>
        <div className="slider__handle" style={{ left: `${position}%` }} />
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={position}
        onChange={(e) => setPosition(Number(e.target.value))}
        aria-label={`Reveal ${rightLabel} vs ${leftLabel}`}
        style={{ width: "100%", marginTop: "0.75rem" }}
      />
    </div>
  );
}
