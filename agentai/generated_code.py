import re
from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://lawyersaathi.com/")
    page.get_by_role("button", name="Get Started Free").click()
    page.get_by_placeholder("Email Address").click()
    page.get_by_placeholder("Email Address").fill("fsjngldn")
    page.get_by_placeholder("Email Address").click()
    page.get_by_placeholder("Email Address").click()
    page.get_by_placeholder("Email Address").fill("fsjngldnlfknglffsd")
    page.get_by_placeholder("Password").click()
    page.get_by_placeholder("Password").fill("fdlksng")

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
