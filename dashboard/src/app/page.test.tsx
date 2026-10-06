// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import LandingPage from "./page";

afterEach(() => cleanup());

describe("LandingPage", () => {
  it("introduces the product and links both forecast actions to the workspace", () => {
    render(<LandingPage />);
    expect(screen.getByRole("heading", { level: 1, name: /Before you.*hit publish.*see the possibilities/ })).toBeTruthy();
    const actions = screen.getAllByRole("link", { name: "Create a forecast" });
    expect(actions).toHaveLength(2);
    actions.forEach((action) => expect(action.getAttribute("href")).toBe("/forecast"));
    expect(screen.queryByRole("textbox", { name: /Video title/ })).toBeNull();
    expect(screen.getByRole("link", { name: /Meet the dataset/ }).getAttribute("href")).toBe("/dataset");
    expect(screen.getByRole("link", { name: /Explore what works/ }).getAttribute("href")).toBe("/insights");
  });

  it("lets visitors explore clearly labelled example values across all four checkpoints", () => {
    render(<LandingPage />);
    for (const [day, views] of [["07", "2,840"], ["14", "4,320"], ["21", "5,790"], ["30", "7,240"]]) {
      const checkpoint = screen.getByRole("button", { name: `DAY ${day}` });
      fireEvent.click(checkpoint);
      expect(checkpoint.getAttribute("aria-pressed")).toBe("true");
      expect(screen.getByText(views)).toBeTruthy();
      expect(screen.getByRole("img", { name: new RegExp(`Day ${Number(day)} selected`) })).toBeTruthy();
      expect(screen.getAllByRole("button").filter((button) => button.getAttribute("aria-pressed") === "true")).toHaveLength(1);
    }
    expect(screen.getByText("Illustrative example")).toBeTruthy();
    expect(screen.getByText("Example numbers, not a real prediction.")).toBeTruthy();
  });
});
