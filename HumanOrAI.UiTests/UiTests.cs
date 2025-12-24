using NUnit.Framework;
using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;
using OpenQA.Selenium.Support.UI;
using SeleniumExtras.WaitHelpers;

namespace HumanOrAI.UiTests;

[TestFixture]
public class UiTests
{
    private IWebDriver _driver = null!;
    private WebDriverWait _wait = null!;

    private const string BaseUrl = "http://localhost:5173";

    [SetUp]
    public void Setup()
    {
        var options = new ChromeOptions();
        options.AddArgument("--window-size=1400,900");

        _driver = new ChromeDriver(options);
        _wait = new WebDriverWait(_driver, TimeSpan.FromSeconds(25));
    }

    [TearDown]
    public void TearDown()
    {
        _driver.Quit();
        _driver.Dispose();
    }

    // ✅ TEST-1: Predict sayfası render (input + button)
    [Test]
    public void PredictPage_ShouldRender_InputAndButton()
    {
        _driver.Navigate().GoToUrl($"{BaseUrl}/predict");

        var input = _wait.Until(ExpectedConditions.ElementExists(By.CssSelector("[data-testid='predict-input']")));
        var button = _wait.Until(ExpectedConditions.ElementExists(By.CssSelector("[data-testid='predict-button']")));

        Assert.That(input.Displayed, Is.True);
        Assert.That(button.Displayed, Is.True);
    }

    // ✅ TEST-2: Boş metin -> alert "Metin boş olamaz"
    [Test]
    public void Predict_EmptyText_ShouldShowAlert()
    {
        _driver.Navigate().GoToUrl($"{BaseUrl}/predict");

        var button = _wait.Until(ExpectedConditions.ElementToBeClickable(By.CssSelector("[data-testid='predict-button']")));
        button.Click();

        // JS alert yakala
        var alert = _wait.Until(ExpectedConditions.AlertIsPresent());
        Assert.That(alert.Text, Does.Contain("Metin boş olamaz"));

        alert.Accept();
    }

    // ✅ TEST-3: Metin gir -> sonuç kartı görünür
    [Test]
    public void Predict_WithText_ShouldShowResultCard()
    {
        _driver.Navigate().GoToUrl($"{BaseUrl}/predict");

        var input = _wait.Until(ExpectedConditions.ElementIsVisible(By.CssSelector("[data-testid='predict-input']")));
        input.SendKeys("This is a scientific abstract about machine learning and classification.");

        var button = _driver.FindElement(By.CssSelector("[data-testid='predict-button']"));
        button.Click();

        // sonuç gelene kadar bekle
        var resultCard = _wait.Until(ExpectedConditions.ElementIsVisible(By.CssSelector("[data-testid='predict-result']")));
        var finalText = _wait.Until(ExpectedConditions.ElementIsVisible(By.CssSelector("[data-testid='predict-final']"))).Text;

        Assert.That(resultCard.Displayed, Is.True);
        Assert.That(finalText, Does.Contain("Sonuç:"));
    }
}
