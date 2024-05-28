from selenium import webdriver  # seems like one needs to pip install selenium to have it work in an IDE
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.firefox.options import Options


def get_reviews_from_soup(soup):
    reviews_container = soup.find_all("ol", attrs={"data-bv-v": "contentItemCollection:2"})[0]

    # authors_container = reviews_container.find_all('span', attrs={'class': 'bv-author'})
    # dates_container = reviews_container.find_all('span', attrs={'class': 'bv-content-datetime-stamp'})
    # scores_container = reviews_container.find_all('span', attrs={'itemprop': 'reviewRating'})
    # texts_container = reviews_container.find_all('div', attrs={'class': 'bv-content-summary-body-text'})
    # helpful_container = reviews_container.find_all('button', attrs={'class': 'bv-content-btn-feedback-yes'})
    # not_helpful_container = reviews_container.find_all('button', attrs={'class': 'bv-content-btn-feedback-no'})
    #
    # if not(len(authors_container) == len(dates_container) == len(scores_container) == len(texts_container) == len(helpful_container) == len(not_helpful_container)):
    #     print('Error:')
    #     print(str(len(authors_container)) + ' authors.')
    #     print(str(len(dates_container)) + ' dates.')
    #     print(str(len(scores_container)) + ' scores.')
    #     print(str(len(texts_container)) + ' texts.')
    #     print(str(len(helpful_container)) + ' helpful votes.')
    #     print(str(len(not_helpful_container)) + ' not helpful votes.')

    # n = len(authors_container)
    authors, dates, scores, texts, helpful, not_helpful = [], [], [], [], [], []

    reviews_list = reviews_container.find_all('li', attrs={'itemprop': 'review'}, recursive=False)
    for item in reviews_list:
        core = item.find('div', attrs={'class': 'bv-content-core'})
        # feedback = item.find('div', attrs={'class': 'bv-active-feedback'})
        authors.append(core.find('div', attrs={'class': 'bv-content-meta'}).find('span', attrs={'class': 'bv-author'}).find('span', attrs={'itemprop': 'name'}).text)
        dates.append(core.find('div', attrs={'class': 'bv-content-meta'}).find('div', attrs={'class': 'bv-content-datetime'}).find('span', attrs={'class': 'bv-content-datetime-stamp'}).text.removesuffix(' \xa0'))
        scores.append(core.find('span', attrs={'class': 'bv-content-rating'}).find('meta', attrs={'itemprop': 'ratingValue'}).get('content', default=None))

        t_container = core.find('div', attrs={'class': 'bv-content-summary-body-text'}).find_all('p')
        t = ''
        for j in range(len(t_container)):
            t = t + t_container[j].text + ' '
        texts.append(t)

        # helpful.append(feedback.find('button', attrs={'class': 'bv-content-btn-feedback-yes'}).find('span', attrs='bv-content-btn-count').text)
        # not_helpful.append(feedback.find('button', attrs={'class': 'bv-content-btn-feedback-no'}).find('span', attrs='bv-content-btn-count').text)

    # for i in range(n):
    #     authors.append(authors_container[i].find('span', attrs={'itemprop': 'name'}).text)
    #     dates.append(dates_container[i].text)
    #     scores.append(scores_container[i].find('meta', attrs={'itemprop': 'ratingValue'}).get('content', default=None))
    #
    #     container = texts_container[i].find_all('p')
    #     t = ''
    #     for j in range(len(container)):
    #         t = t + container[j].text + ' '
    #     texts.append(t)
    #
    #     helpful.append(helpful_container[i].find('span', attrs='bv-content-btn-count').text)
    #     not_helpful.append(not_helpful_container[i].find('span', attrs='bv-content-btn-count').text)

    return authors, dates, scores, texts, helpful, not_helpful


def get_reviews_from_url(url):
    all_authors, all_dates, all_scores, all_texts, all_helpful, all_not_helpful = [], [], [], [], [], []

    more_selector = "#nav-pdp-tab-header-13 > div.row.view-more-v2 > div > input"

    driver = webdriver.Firefox()
    driver.implicitly_wait(30)
    driver.get(url)

    timeout = 30
    try:
        more_clickable = EC.element_to_be_clickable((By.CSS_SELECTOR, more_selector))
        WebDriverWait(driver, timeout).until(more_clickable)
    except TimeoutException:
        print("Timed out waiting for page to load")

    more_button = driver.find_element(By.CSS_SELECTOR, more_selector)
    more_button.location_once_scrolled_into_view
    more_button.click()  # click load more button

    page_exists = True

    while page_exists:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        authors, dates, scores, texts, helpful, not_helpful = get_reviews_from_soup(soup)

        all_authors = all_authors + authors
        all_dates = all_dates + dates
        all_scores = all_scores + scores
        all_texts = all_texts + texts
        all_helpful = all_helpful + helpful
        all_not_helpful = all_not_helpful + not_helpful

        # Finding page buttons
        pagination = soup.find('ul', attrs={'class': 'bv-content-pagination-buttons'})
        if pagination:
            next_page = pagination.find_all('li')[1]
            if next_page.find('a'):
                next_selector = '#BVRRContainer > div > div > div > div > div.bv-content-pagination > div > ul > li.bv-content-pagination-buttons-item.bv-content-pagination-buttons-item-next > a'
                next_clickable = EC.element_to_be_clickable((By.CSS_SELECTOR, next_selector))
                WebDriverWait(driver, timeout).until(next_clickable)
                next_button = driver.find_element(By.CSS_SELECTOR, next_selector)
                next_button.click()
            else:
                page_exists = False
        else:
            page_exists = False

    return all_authors, all_dates, all_scores, all_texts, all_helpful, all_not_helpful

# url = 'https://www.costco.com/macbook-air-laptop-13.6-inch---apple-m2-chip%2c-8-core-cpu%2c-8-core-gpu%2c-8gb-memory%2c-256gb-ssd-storage.product.100713212.html'
# all_authors, all_dates, all_scores, all_texts, all_helpful, all_not_helpful = get_reviews_from_url(url)

# 'https://www.costco.com/kirkland-signature%2c-organic-extra-virgin-olive-oil%2c-2-l.product.100334841.html',
#              'https://www.costco.com/clear-touch-food-prep-poly-gloves%2c-one-size%2c-2%2c000-count.product.100410420.html',
#              'https://www.costco.com/frigidaire-stainless-steel-bottom-loading-water-cooler.product.100493343.html',
#              'https://www.costco.com/mr.-coffee-one-touch-coffeehouse-espresso-and-cappuccino-machine%2c-dark-stainless.product.100688290.html',
#              'https://www.costco.com/cuisinart-custom-select-4-slice-toaster.product.100774095.html',
#              'https://www.costco.com/hotel-signature-800-thread-count-cotton-6-piece-sheet-set.product.4000099225.html',
#              'https://www.costco.com/ecovacs-deebot-neo%2b-vacuum-and-mop-robot-with-auto-empty-station.product.4000186722.html',
#              'https://www.costco.com/neocube-50-liter-dual-compartment-28-liter-and-18-liter-stainless-steel-recycle-and-trash-bin.product.100514717.html',
# 'https://www.costco.com/beats-studio-buds-%2b-true-wireless-noise-cancelling-earbuds-with-applecare%2b-included.product.4000183124.html',
#              'https://www.costco.com/kirkland-signature-organic-sumatra-whole-bean-coffee%2c-2-lbs%2c-2-pack.product.100787428.html',
#              'https://www.costco.com/kirkland-signature-stainless-steel-6-burner-gas-grill.product.4000098457.html',

urls_list = ['https://www.costco.com/kirkland-signature-healthy-weight-formula-chicken-%2526-vegetable-dog-food-40-lb..product.100343450.html',
             'https://www.costco.com/macbook-air-laptop-13.6-inch---apple-m2-chip%2c-8-core-cpu%2c-8-core-gpu%2c-8gb-memory%2c-256gb-ssd-storage.product.100713212.html']

all_authors, all_dates, all_scores, all_texts, all_helpful, all_not_helpful = [], [], [], [], [], []
items = []

for url in urls_list:
    print('Scraping ' + url)
    authors, dates, scores, texts, helpful, not_helpful = get_reviews_from_url(url)
    all_authors = all_authors + authors
    all_dates = all_dates + dates
    all_scores = all_scores + scores
    all_texts = all_texts + texts
    all_helpful = all_helpful + helpful
    all_not_helpful = all_not_helpful + not_helpful
    for i in range(len(all_authors)):
        items.append(url)
