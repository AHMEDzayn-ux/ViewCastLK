// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import ForecastForm from "./ForecastForm";

vi.mock("@/lib/api/forecast", () => ({
  isChannelLookupMockMode: () => false,
  lookupChannelStats: vi.fn(),
  PredictionApiError: class PredictionApiError extends Error {},
}));

afterEach(() => {
  cleanup();
  window.sessionStorage.clear();
});

describe("ForecastForm channel selection", () => {
  it("uses the connected YouTube channel without asking for its handle", () => {
    const onSubmit = vi.fn();
    render(
      <ForecastForm
        onSubmit={onSubmit}
        onReset={vi.fn()}
        isLoading={false}
        canLookupChannel={true}
        connectedChannel={{ id: "UC-connected", title: "My channel" }}
        isCheckingChannel={false}
      />,
    );

    expect(screen.getByText("My channel")).toBeTruthy();
    expect(screen.queryByRole("textbox", { name: "YouTube channel" })).toBeNull();

    fireEvent.change(screen.getByRole("textbox", { name: "Video title" }), {
      target: { value: "My next video" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "Video category" }), {
      target: { value: "Education" },
    });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Planned duration Minutes" }), {
      target: { value: "5" },
    });
    fireEvent.click(screen.getByRole("radio", { name: "Standard video" }));
    fireEvent.change(screen.getByRole("combobox", { name: "Audio language" }), {
      target: { value: "English" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate forecast" }));

    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      channelIdentifier: "UC-connected",
    }));
  });

  it("keeps manual channel entry for an account without a YouTube connection", () => {
    render(
      <ForecastForm
        onSubmit={vi.fn()}
        onReset={vi.fn()}
        isLoading={false}
        canLookupChannel={true}
        connectedChannel={null}
        isCheckingChannel={false}
      />,
    );

    expect(screen.getByRole("textbox", { name: "YouTube channel" })).toBeTruthy();
  });
});
