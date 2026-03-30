#!/usr/bin/env ts-node
/**
 * main.tsx --- browser entrypoint that mounts the dashboard root component
 *
 * Contains:
 *   root render: mounts <App /> into the #root element from index.html
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import "./styles.css";

const rootElement = document.getElementById("root");

if (rootElement === null) {
  throw new Error("index.html is missing the #root mount point");
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
