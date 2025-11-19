# Determinants of Health-Seeking Behavior in India

## 📌 Project Overview

This project analyzes data from the National Family Health Survey (NFHS) to identify the primary drivers of health-seeking behavior among women in India.

Using Logistic Regression and Feature Importance analysis, we predict the likelihood of a woman visiting a health facility (s368). The model challenges the traditional assumption that "Wealth" is the sole predictor of health access, revealing that Information Access (Media) and Intra-Household Dynamics play a more decisive role.

## 🔗 Quick Links

* 📄 Full Research Report: [View Google Doc](https://docs.google.com/document/d/1qso6DaaMGJEoE5dTEzYodNpQxgnZhm0-OZagi48KtqA/edit?usp=sharing)

* 💻 Interactive Analysis: [Run on Google Colab](https://colab.research.google.com/drive/1HWqYmmqANCzHLA3V83gS9KqC-Gvdcay3?usp=sharing)

## 🎯 Objective

To determine which socio-demographic features most strongly influence a woman's decision to seek medical care, specifically distinguishing between:

* Predisposing Factors: Age, Family Size.

* Enabling Factors: Media Exposure, Insurance.

* Need Factors: (Implicit drivers).

## 🔍 Key Insights & Findings

1. The "Information Dividend" (Media > Wealth)

    Our model identifies Media Exposure as the strongest positive driver of health-seeking behavior, often outperforming direct wealth assets.

* Data: v159 (TV Frequency) and v157 (Newspaper/Magazine Frequency) are top positive predictors.

* Implication: This supports the Knowledge Gap Hypothesis. Women with high media engagement possess higher health literacy, acting as a powerful "Enabling Factor" (per Andersen's Behavioral Model).

2. The "Resource Dilution" Hypothesis

    While total household size is a positive safety net, the composition of the household reveals a negative trend.

* Data: hv010 (Count of Eligible Women) has a significant negative coefficient.

* Implication: In large households, multiple women compete for finite resources (time, money, chaperones). This "crowding out" effect reduces the individual likelihood of accessing care.

3. The Sanitation Class Divide

    Correlation analysis reveals that sanitation infrastructure is the most rigid proxy for economic status.

* Data: hv205 (Toilet Type) has a stronger correlation with wealth than education.

* Insight: Unlike mobile phones (which show market saturation), the type of toilet facility remains a distinct separator of economic class.

4. The "Distress Asset" Paradox

* Data: hv227 (Mosquito Bed Net) shows a negative correlation with wealth.

* Insight: Unlike "Aspirational Assets" (TVs), bed nets function as "Distress Assets." Ownership is higher in vulnerable, lower-income households (often due to aid distribution or lack of sealed housing).

## 📚 References & Theoretical Frameworks

This analysis is grounded in:

* Andersen’s Behavioral Model of Health Services Use (Predisposing vs. Enabling factors).

* Resource Dilution Theory (Intra-household competition).

* Jensen & Oster (NBER): The Power of TV: Cable Television and Women's Status in India.


