"""Synthetic enterprise corpus for "Harbor Fresh Markets" (a fictional grocery retailer).

Every document is invented for demo purposes. Each entry: filename, format, title, sections.
The golden evaluation set (eval/golden_set.jsonl) is generated from the `qa` facts below.
"""

DOCS = [
    {
        "file": "product_recall_policy", "fmt": "pdf", "title": "Product Recall Policy",
        "sections": [
            ("Purpose", "This policy defines how Harbor Fresh Markets removes unsafe products from sale. "
             "It applies to all 42 stores and both distribution centers."),
            ("Recall Classes", "A Class I recall involves a reasonable probability of serious health consequences. "
             "Class I products must be pulled from shelves within 2 hours of notification. "
             "A Class II recall must be completed within 24 hours. Class III recalls must be completed within 72 hours."),
            ("Current Recalls", "As of June 2026 the following SKUs are under active recall: "
             "SKU HF-1042 Organic Baby Spinach 10oz (Class I, possible Listeria), "
             "SKU HF-2210 Almond Butter 16oz (Class II, undeclared peanut), and "
             "SKU HF-3307 Frozen Mango Chunks 32oz (Class II, possible Hepatitis A)."),
            ("Customer Refunds", "Customers returning recalled items receive a full refund without a receipt. "
             "Refunds for recalled items are issued in the original tender when possible, otherwise as store credit."),
        ],
        "qa": [
            ("How quickly must a Class I recalled product be pulled from shelves?", "within 2 hours of notification", "Recall Classes"),
            ("Which SKUs are under active recall?", "HF-1042, HF-2210 and HF-3307", "Current Recalls"),
            ("Do customers need a receipt to return a recalled item?", "No, full refund without a receipt", "Customer Refunds"),
            ("What is the deadline to complete a Class III recall?", "72 hours", "Recall Classes"),
        ],
    },
    {
        "file": "return_and_refund_policy", "fmt": "docx", "title": "Return and Refund Policy",
        "sections": [
            ("General Returns", "Most unopened non-perishable items can be returned within 30 days with a receipt. "
             "Without a receipt, customers receive store credit at the lowest selling price in the past 60 days."),
            ("Perishables", "Perishable items such as produce, dairy and bakery can be returned within 7 days "
             "if the customer is dissatisfied with quality."),
            ("Exclusions", "Alcohol, tobacco, gift cards, prescription medications and lottery tickets cannot be returned."),
            ("Manager Override", "A store manager may approve returns up to $150 outside the policy window. "
             "Returns above $150 outside policy require district manager approval."),
        ],
        "qa": [
            ("How many days do customers have to return non-perishable items with a receipt?", "30 days", "General Returns"),
            ("Can gift cards be returned?", "No, gift cards cannot be returned", "Exclusions"),
            ("Up to what amount can a store manager approve a return outside policy?", "$150", "Manager Override"),
            ("What is the return window for perishable items?", "7 days", "Perishables"),
        ],
    },
    {
        "file": "food_safety_temperature_standards", "fmt": "pdf", "title": "Food Safety Temperature Standards",
        "sections": [
            ("Cold Holding", "Refrigerated cases must hold products at 41 degrees Fahrenheit or below. "
             "Frozen cases must hold products at 0 degrees Fahrenheit or below."),
            ("Hot Holding", "Hot foods in the deli must be held at 135 degrees Fahrenheit or above."),
            ("Temperature Logs", "Associates must record case temperatures every 4 hours in the digital log. "
             "Any reading out of range for more than 30 minutes requires the product to be discarded."),
            ("Cooling", "Cooked foods must cool from 135F to 70F within 2 hours and from 70F to 41F within the next 4 hours."),
        ],
        "qa": [
            ("What temperature must refrigerated cases hold products at?", "41 degrees Fahrenheit or below", "Cold Holding"),
            ("How often must case temperatures be logged?", "every 4 hours", "Temperature Logs"),
            ("What is the minimum hot holding temperature in the deli?", "135 degrees Fahrenheit", "Hot Holding"),
        ],
    },
    {
        "file": "employee_handbook_attendance", "fmt": "docx", "title": "Employee Handbook - Attendance and Scheduling",
        "sections": [
            ("Scheduling", "Schedules are posted every Thursday for the week starting 10 days later. "
             "Associates may swap shifts through the StaffHub app with manager approval."),
            ("Call Outs", "Associates must call out at least 2 hours before a scheduled shift by contacting the store hotline."),
            ("Attendance Points", "An unexcused absence earns 1 point and a no-call no-show earns 3 points. "
             "Reaching 8 points in a rolling 12 months results in termination review."),
            ("Breaks", "Associates working 6 or more hours receive an unpaid 30-minute meal break and two paid 15-minute breaks."),
        ],
        "qa": [
            ("How many attendance points does a no-call no-show earn?", "3 points", "Attendance Points"),
            ("How far in advance must associates call out?", "at least 2 hours before the shift", "Call Outs"),
            ("When are schedules posted?", "every Thursday", "Scheduling"),
        ],
    },
    {
        "file": "paid_time_off_policy", "fmt": "md", "title": "Paid Time Off Policy",
        "sections": [
            ("Accrual", "Full-time associates accrue 1 hour of PTO for every 26 hours worked, up to 120 hours per year. "
             "Part-time associates accrue 1 hour for every 40 hours worked."),
            ("Carryover", "Up to 40 hours of unused PTO carries over into the next calendar year."),
            ("Requests", "PTO requests must be submitted at least 14 days in advance for planned time off."),
        ],
        "qa": [
            ("What is the maximum PTO a full-time associate can accrue per year?", "120 hours", "Accrual"),
            ("How much unused PTO carries over?", "up to 40 hours", "Carryover"),
            ("How far in advance must planned PTO be requested?", "14 days", "Requests"),
        ],
    },
    {
        "file": "cash_handling_procedures", "fmt": "pdf", "title": "Cash Handling Procedures",
        "sections": [
            ("Till Limits", "Register tills must not exceed $300 in cash. Excess cash must be dropped into the safe via a skim."),
            ("Counterfeit Detection", "Bills of $50 and $100 must be checked with a detection pen and the UV light."),
            ("Variances", "A till variance greater than $5 must be documented. Three variances over $20 within 90 days "
             "trigger a coaching conversation."),
        ],
        "qa": [
            ("What is the maximum cash allowed in a register till?", "$300", "Till Limits"),
            ("Which bills must be checked for counterfeits?", "$50 and $100 bills", "Counterfeit Detection"),
        ],
    },
    {
        "file": "loyalty_program_terms", "fmt": "docx", "title": "Harbor Rewards Loyalty Program Terms",
        "sections": [
            ("Earning Points", "Members earn 1 point per $1 spent. Pharmacy purchases and alcohol do not earn points."),
            ("Redemption", "Every 100 points can be redeemed for $1 off. Points expire 12 months after they are earned."),
            ("Fuel Rewards", "Members can redeem 200 points for 10 cents off per gallon at partner fuel stations, up to 20 gallons."),
            ("Tiers", "Gold tier members, who spend over $3,000 per year, earn 1.5 points per $1."),
        ],
        "qa": [
            ("How many points are needed for $1 off?", "100 points", "Redemption"),
            ("When do loyalty points expire?", "12 months after they are earned", "Redemption"),
            ("How much must a member spend to reach Gold tier?", "over $3,000 per year", "Tiers"),
        ],
    },
    {
        "file": "vendor_onboarding_guide", "fmt": "pdf", "title": "Vendor Onboarding Guide",
        "sections": [
            ("Requirements", "New vendors must provide a certificate of insurance with at least $2 million general liability "
             "coverage and a current third-party food safety audit (SQF or BRC)."),
            ("Payment Terms", "Standard payment terms are Net 45. Local farm vendors are paid Net 15."),
            ("EDI", "Vendors shipping more than 50 cases per week must support EDI 850 purchase orders and EDI 856 ship notices."),
        ],
        "qa": [
            ("What are the standard vendor payment terms?", "Net 45", "Payment Terms"),
            ("How much liability insurance must new vendors carry?", "$2 million general liability", "Requirements"),
        ],
    },
    {
        "file": "information_security_policy", "fmt": "md", "title": "Information Security Policy",
        "sections": [
            ("Passwords", "Passwords must be at least 14 characters and are rotated every 180 days. "
             "Multi-factor authentication is required for all remote access."),
            ("Data Classification", "Data is classified as Public, Internal, Confidential or Restricted. "
             "Customer payment data is Restricted and must never be stored on local devices."),
            ("Incident Reporting", "Suspected security incidents must be reported to the Security Operations Center within 1 hour."),
        ],
        "qa": [
            ("What is the minimum password length?", "14 characters", "Passwords"),
            ("How quickly must security incidents be reported?", "within 1 hour", "Incident Reporting"),
        ],
    },
    {
        "file": "store_opening_checklist", "fmt": "docx", "title": "Store Opening Checklist",
        "sections": [
            ("Before Opening", "The opening manager arrives 60 minutes before opening, disarms the alarm, and walks the sales floor."),
            ("Registers", "At least 4 registers must be open at opening time, with self-checkout lanes powered on 15 minutes early."),
            ("Fresh Departments", "Produce displays must be fully stocked and misted before doors open at 7:00 AM."),
        ],
        "qa": [
            ("How early does the opening manager arrive?", "60 minutes before opening", "Before Opening"),
            ("How many registers must be open at opening?", "at least 4", "Registers"),
        ],
    },
    {
        "file": "pharmacy_operations_manual", "fmt": "pdf", "title": "Pharmacy Operations Manual",
        "sections": [
            ("Controlled Substances", "Schedule II inventory is counted daily and reconciled weekly by the pharmacist in charge."),
            ("Immunizations", "Pharmacists may administer flu, COVID-19 and shingles vaccines to patients aged 18 and older "
             "without a prescription."),
            ("Hours", "Pharmacies operate from 9:00 AM to 9:00 PM on weekdays and 10:00 AM to 6:00 PM on weekends."),
        ],
        "qa": [
            ("How often is Schedule II inventory counted?", "daily", "Controlled Substances"),
            ("What are pharmacy weekday hours?", "9:00 AM to 9:00 PM", "Hours"),
        ],
    },
    {
        "file": "sustainability_report_2025", "fmt": "pdf", "title": "Sustainability Report 2025",
        "sections": [
            ("Food Waste", "In 2025 Harbor Fresh Markets diverted 78 percent of food waste from landfill through donation and composting."),
            ("Energy", "Twelve stores now have rooftop solar, supplying about 22 percent of their electricity."),
            ("Goals", "The company targets zero food waste to landfill by 2030 and a 50 percent cut in refrigerant emissions by 2028."),
        ],
        "qa": [
            ("What percentage of food waste was diverted from landfill in 2025?", "78 percent", "Food Waste"),
            ("How many stores have rooftop solar?", "twelve", "Energy"),
        ],
    },
    {
        "file": "pricing_and_promotions_guide", "fmt": "docx", "title": "Pricing and Promotions Guide",
        "sections": [
            ("Price Changes", "Shelf tags are updated every Wednesday night to match the weekly ad starting Thursday."),
            ("Price Match", "Stores match advertised prices from local competitors within 10 miles for identical items."),
            ("Scan Guarantee", "If an item scans higher than the shelf price, the customer receives the first item free, up to $10."),
        ],
        "qa": [
            ("What happens if an item scans higher than the shelf price?", "first item free, up to $10", "Scan Guarantee"),
            ("When are shelf tags updated?", "every Wednesday night", "Price Changes"),
        ],
    },
    {
        "file": "delivery_and_pickup_service", "fmt": "md", "title": "Delivery and Curbside Pickup Service",
        "sections": [
            ("Fees", "Curbside pickup is free on orders over $35; otherwise a $4.95 fee applies. Delivery costs $7.95."),
            ("Substitutions", "Shoppers substitute out-of-stock items with the closest match unless the customer opts out."),
            ("Windows", "Pickup windows are one hour long, and orders must be placed at least 3 hours before the window."),
        ],
        "qa": [
            ("When is curbside pickup free?", "on orders over $35", "Fees"),
            ("How much does delivery cost?", "$7.95", "Fees"),
        ],
    },
    {
        "file": "workplace_safety_program", "fmt": "pdf", "title": "Workplace Safety Program",
        "sections": [
            ("Injury Reporting", "All workplace injuries must be reported to a manager before the end of the shift."),
            ("Equipment", "Only certified associates aged 18 or older may operate balers, compactors and powered pallet jacks."),
            ("Spills", "Spills must be guarded immediately and cleaned within 5 minutes using the spill station kit."),
        ],
        "qa": [
            ("How soon must spills be cleaned?", "within 5 minutes", "Spills"),
            ("Who may operate balers and compactors?", "certified associates aged 18 or older", "Equipment"),
        ],
    },
    {
        "file": "inventory_management_procedures", "fmt": "docx", "title": "Inventory Management Procedures",
        "sections": [
            ("Cycle Counts", "Each department completes cycle counts weekly, and full physical inventory happens twice a year in January and July."),
            ("Shrink", "Shrink above 2.5 percent of sales in any department triggers a loss prevention review."),
            ("Replenishment", "Automated replenishment orders are generated nightly when on-hand units fall below the reorder point."),
        ],
        "qa": [
            ("When does full physical inventory take place?", "twice a year, in January and July", "Cycle Counts"),
            ("What shrink level triggers a loss prevention review?", "above 2.5 percent of sales", "Shrink"),
        ],
    },
    {
        "file": "customer_service_standards", "fmt": "md", "title": "Customer Service Standards",
        "sections": [
            ("Greeting", "Associates greet customers within 10 feet and make eye contact within 5 feet."),
            ("Complaints", "Customer complaints must receive a response within 24 hours and be resolved within 3 business days."),
            ("Checkout", "When more than 3 customers are waiting in a line, an additional register must be opened."),
        ],
        "qa": [
            ("How quickly must customer complaints get a response?", "within 24 hours", "Complaints"),
            ("When must an additional register be opened?", "when more than 3 customers are waiting", "Checkout"),
        ],
    },
    {
        "file": "allergen_labeling_policy", "fmt": "pdf", "title": "Allergen Labeling Policy",
        "sections": [
            ("Major Allergens", "Store-prepared foods must declare the nine major allergens: milk, eggs, fish, shellfish, "
             "tree nuts, peanuts, wheat, soybeans and sesame."),
            ("Label Errors", "Undeclared allergens are treated as a Class II recall at minimum and escalated to Quality Assurance."),
        ],
        "qa": [
            ("How many major allergens must be declared?", "nine", "Major Allergens"),
            ("How are undeclared allergens treated?", "as a Class II recall at minimum", "Label Errors"),
        ],
    },
    {
        "file": "travel_and_expense_policy", "fmt": "docx", "title": "Travel and Expense Policy",
        "sections": [
            ("Meals", "The daily meal allowance for business travel is $65. Alcohol is not reimbursable."),
            ("Mileage", "Personal vehicle use is reimbursed at the IRS standard mileage rate."),
            ("Submission", "Expense reports must be submitted within 30 days of the trip with itemized receipts for any expense over $25."),
        ],
        "qa": [
            ("What is the daily meal allowance for business travel?", "$65", "Meals"),
            ("How soon must expense reports be submitted?", "within 30 days of the trip", "Submission"),
        ],
    },
    {
        "file": "emergency_response_plan", "fmt": "pdf", "title": "Emergency Response Plan",
        "sections": [
            ("Hurricanes", "Stores in hurricane zones close 6 hours before projected tropical-storm-force winds arrive."),
            ("Power Outage", "If power is lost for more than 4 hours, refrigerated and frozen products must be evaluated by Quality Assurance."),
            ("Evacuation", "Assembly points are posted at each exit; managers perform a headcount within 10 minutes of evacuation."),
        ],
        "qa": [
            ("How long can power be out before refrigerated products need evaluation?", "more than 4 hours", "Power Outage"),
            ("When do stores close before a hurricane?", "6 hours before tropical-storm-force winds", "Hurricanes"),
        ],
    },
    {
        "file": "private_label_quality_standards", "fmt": "md", "title": "Private Label Quality Standards",
        "sections": [
            ("Testing", "Every Harbor Select private label product is taste-tested against the national brand leader before launch "
             "and must win or tie in at least 60 percent of blind tests."),
            ("Supplier Audits", "Private label suppliers are audited annually, and a score below 85 requires a corrective action plan."),
        ],
        "qa": [
            ("What blind-test win rate must private label products achieve?", "at least 60 percent", "Testing"),
            ("What supplier audit score requires a corrective action plan?", "below 85", "Supplier Audits"),
        ],
    },
]

PRODUCTS = [
    ("HF-1042", "Organic Baby Spinach 10oz", "Produce", 3.99),
    ("HF-1100", "Gala Apples 3lb", "Produce", 4.49),
    ("HF-1205", "Bananas per lb", "Produce", 0.59),
    ("HF-2210", "Almond Butter 16oz", "Grocery", 8.99),
    ("HF-2301", "Rolled Oats 42oz", "Grocery", 5.29),
    ("HF-2455", "Harbor Select Pasta 16oz", "Grocery", 1.49),
    ("HF-3307", "Frozen Mango Chunks 32oz", "Frozen", 9.49),
    ("HF-3320", "Frozen Pizza Margherita", "Frozen", 6.99),
    ("HF-4010", "Whole Milk 1gal", "Dairy", 3.79),
    ("HF-4088", "Greek Yogurt 32oz", "Dairy", 5.99),
    ("HF-5001", "Sourdough Loaf", "Bakery", 4.99),
    ("HF-6120", "Rotisserie Chicken", "Deli", 7.99),
]

STORES = [
    (101, "Harbor Point", "Tampa", "West"),
    (102, "Bayshore", "Tampa", "West"),
    (201, "Lakeside", "Orlando", "Central"),
    (202, "Winter Park", "Orlando", "Central"),
    (301, "Riverside", "Jacksonville", "North"),
    (302, "San Marco", "Jacksonville", "North"),
]
